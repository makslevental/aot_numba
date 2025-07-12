import functools
import os
import subprocess
import tempfile
from pathlib import Path

from llvmlite.binding import Linkage
from numba.core import config
from numba.core.codegen import CPUCodeLibrary
from numba.core.compiler import compile_extra, Flags
from numba.core.compiler_lock import global_compiler_lock
from numba.core.funcdesc import FunctionDescriptor, default_mangler, qualifying_prefix
from numba.core.runtime import nrtdynmod
from numba.core.runtime.nrtopt import remove_redundant_nrt_refct
from numba.pycc import CC
from numba.pycc.compiler import ModuleCompiler, _ModuleCompiler
from numba.pycc.platform import Toolchain, CCompiler

HERE = Path(__file__).parent


def _mangle_method_symbol(self, func_name):
    return func_name


@global_compiler_lock
def _cull_exports(self):
    """Read all the exported functions/modules in the translator
    environment, and join them into a single LLVM module.
    """
    self.exported_function_types = {}
    self.function_environments = {}
    self.environment_gvs = {}

    codegen = self.context.codegen()
    library = codegen.create_library(self.module_name)

    # Generate IR for all exported functions
    flags = Flags()
    flags.no_compile = True
    if not self.export_python_wrap:
        flags.no_cpython_wrapper = True
        flags.no_cfunc_wrapper = True
    if self.use_nrt:
        flags.nrt = True
        # Compile NRT helpers
        nrt_module, _ = nrtdynmod.create_nrt_module(self.context)
        library.add_ir_module(nrt_module)

    for entry in self.export_entries:
        cres = compile_extra(
            self.typing_context,
            self.context,
            entry.function,
            entry.signature.args,
            entry.signature.return_type,
            flags,
            locals={},
            library=library,
        )

        func_name = cres.fndesc.llvm_func_name
        llvm_func = cres.library.get_function(func_name)

        if self.export_python_wrap:
            # llvm_func.linkage = "internal"
            llvm_func.linkage = "external"
            wrappername = cres.fndesc.llvm_cpython_wrapper_name
            wrapper = cres.library.get_function(wrappername)
            wrapper.name = self._mangle_method_symbol(entry.symbol)
            wrapper.linkage = "external"
            fnty = cres.target_context.call_conv.get_function_type(
                cres.fndesc.restype, cres.fndesc.argtypes
            )
            self.exported_function_types[entry] = fnty
            self.function_environments[entry] = cres.environment
            self.environment_gvs[entry] = cres.fndesc.env_name
        else:
            llvm_func.name = entry.symbol
            self.dll_exports.append(entry.symbol)

    if self.export_python_wrap:
        wrapper_module = library.create_ir_module("wrapper")
        self._emit_python_wrapper(wrapper_module)
        library.add_ir_module(wrapper_module)

    # Hide all functions in the DLL except those explicitly exported
    library.finalize()
    for fn in library.get_defined_functions():
        if fn.name not in self.dll_exports:
            ################################################################
            # make all symbols have default visibility
            fn.visibility = "default"
            ################################################################
            # if fn.linkage in {Linkage.private, Linkage.internal}:
            #     # Private/Internal linkage must have "default" visibility
            #     fn.visibility = "default"
            # else:
            #     fn.visibility = "hidden"
    return library


_ModuleCompiler._mangle_method_symbol = _mangle_method_symbol
_ModuleCompiler._cull_exports = _cull_exports


def __init__(
    self,
    native,
    modname,
    qualname,
    unique_name,
    doc,
    typemap,
    restype,
    calltypes,
    args,
    kws,
    mangler=None,
    argtypes=None,
    inline=False,
    noalias=False,
    env_name=None,
    global_dict=None,
    abi_tags=(),
    uid=None,
):
    self.native = native
    self.modname = modname
    self.global_dict = global_dict
    self.qualname = qualname
    self.unique_name = unique_name
    self.doc = doc
    # XXX typemap and calltypes should be on the compile result,
    # not the FunctionDescriptor
    self.typemap = typemap
    self.calltypes = calltypes
    self.args = args
    self.kws = kws
    self.restype = restype
    # Argument types
    if argtypes is not None:
        assert isinstance(argtypes, tuple), argtypes
        self.argtypes = argtypes
    else:
        # Get argument types from the type inference result
        # (note the "arg.FOO" convention as used in typeinfer
        self.argtypes = tuple(self.typemap["arg." + a] for a in args)
    mangler = default_mangler if mangler is None else mangler
    # The mangled name *must* be unique, else the wrong function can
    # be chosen at link time.

    ##############################################################
    # Knockout abi tags
    ##############################################################
    qualprefix = qualifying_prefix(self.modname, self.qualname)
    self.uid = uid
    self.mangled_name = mangler(
        qualprefix,
        self.argtypes,
        # abi_tags=abi_tags,
        # uid=uid,
    )
    if env_name is None:
        env_name = mangler(
            ".NumbaEnv.{}".format(qualprefix),
            self.argtypes,
            # abi_tags=abi_tags, uid=uid
        )
    ##############################################################

    self.env_name = env_name
    self.inline = inline
    self.noalias = noalias
    self.abi_tags = abi_tags


FunctionDescriptor.__init__ = __init__


def link_shared_library(
    self,
    output,
    objects,
    libraries=(),
    library_dirs=(),
    export_symbols=(),
    extra_ldflags=None,
):
    """
    Create a shared library *output* linking the given *objects*
    and *libraries* (all strings).
    """
    output_dir, output_filename = os.path.split(output)
    self._compiler.link(
        CCompiler.SHARED_LIBRARY,
        objects,
        output_filename,
        output_dir,
        libraries,
        library_dirs,
        export_symbols=export_symbols,
        extra_preargs=extra_ldflags,
    )


def link_static_library(
    self,
    output,
    objects,
    libraries=(),
    library_dirs=(),
    export_symbols=(),
    extra_ldflags=None,
):
    """
    Create a shared library *output* linking the given *objects*
    and *libraries* (all strings).
    """
    output_dir, output_filename = os.path.split(output)
    self._compiler.create_static_lib(
        objects,
        output_filename.replace(".so", ""),
        output_dir,
        # libraries,
        # library_dirs,
        # export_symbols=export_symbols,
        # extra_preargs=extra_ldflags,
    )


Toolchain.link_shared_library = link_shared_library
Toolchain.link_static_library = link_static_library


class MyCC(CC):

    def __init__(self, extension_name, source_module=None):
        super().__init__(extension_name, source_module)
        name, _, ext = self._output_file.split(".")
        self._output_file = name + "." + ext

    @global_compiler_lock
    def _compile_object_files(self, build_dir):
        compiler = ModuleCompiler(
            self._export_entries,
            self._basename,
            self._use_nrt,
            cpu_name=self._target_cpu,
        )
        compiler.external_init_function = self._init_function
        temp_obj = os.path.join(
            build_dir, os.path.splitext(self._output_file)[0] + ".o"
        )
        # emit header
        compiler.emit_header(str(HERE / self._basename))
        compiler.write_native_object(temp_obj, wrap=True)
        compiler.write_llvm_bitcode(temp_obj + ".bc", wrap=True)
        subprocess.check_call(["llvm-dis-20", temp_obj + ".bc"])
        return [temp_obj], compiler.dll_exports

    @global_compiler_lock
    def compile(self):
        """
        Compile the extension module.
        """
        self._toolchain.verbose = self.verbose
        build_dir = tempfile.mkdtemp(prefix="pycc-build-%s-" % self._basename, dir=HERE)

        # Compile object file
        objects, dll_exports = self._compile_object_files(build_dir)

        # Compile mixins
        objects += self._compile_mixins(build_dir)

        # Then create shared library
        extra_ldflags = self._get_extra_ldflags()
        output_dll = os.path.join(self._output_dir, self._output_file)
        libraries = self._toolchain.get_python_libraries()
        library_dirs = self._toolchain.get_python_library_dirs()
        ##########################################################################
        # build shared library instead of shared object
        # see above
        self._toolchain.link_shared_library(
            output_dll,
            objects,
            libraries,
            library_dirs,
            export_symbols=dll_exports,
            extra_ldflags=extra_ldflags,
        )
        ##########################################################################


cc = MyCC("my_module")
# Uncomment the following line to print out the compilation steps
cc.verbose = True


@cc.export("mult", "f8(f8, f8)")
def mult(a, b):
    return a * b


@cc.export("square", "f8(f8)")
def square(a):
    return a**2


if __name__ == "__main__":
    cc.compile()
