# # #!/usr/bin/env python3
# # """A template for compilers using xDSL."""
# #
# # from xdsl.context import Context
# # from xdsl.dialects import arith, builtin
# #
# #
# # def get_context() -> Context:
# #     """Get a context with the dialects required by the compiler."""
# #     ctx = Context()
# #     ctx.load_dialect(arith.Arith)
# #     ctx.load_dialect(builtin.Builtin)
# #     return ctx
# #
# #
# # if __name__ == "__main__":
# #     raise NotImplementedError("Template doesn't implement anything!")
#
#
# # src/xdsltemplate/compiler.py
#
# from xdsl.context import MLContext
# from xdsl.dialects import builtin, func, memref, arith
# from xdsl.ir import MLIRType
# from xdsl.parser import Parser
# from io import StringIO
#
# # Import your transform
# from xdsltemplate.transforms.memref_to_emitc import MemRefToEmitCPass
#
#
# def run_conversion(input_mlir: str) -> str:
#     """Convert memref to emitc in MLIR code"""
#
#     # Create context and register dialects
#     ctx = MLContext()
#     ctx.load_dialect(builtin.Builtin)
#     ctx.load_dialect(func.Func)
#     ctx.load_dialect(memref.MemRef)
#     ctx.load_dialect(arith.Arith)
#
#     # Load your emitc dialect
#     try:
#         from xdsl.dialects import emitc
#         ctx.load_dialect(emitc.EmitC)
#     except ImportError:
#         from xdsltemplate.dialects import emitc_ptr
#         ctx.load_dialect(emitc_ptr.EmitCPtr)
#
#     # Parse input
#     parser = Parser(ctx, input_mlir)
#     module = parser.parse_module()
#
#     # Apply conversion pass
#     pass_instance = MemRefToEmitCPass()
#     pass_instance.apply(ctx, module)
#
#     # Print result
#     output = StringIO()
#     from xdsl.printer import Printer
#     printer = Printer(stream=output)
#     printer.print(module)
#
#     return output.getvalue()
#
#
# if __name__ == "__main__":
#     test_input = """
#     func.func @foo(%A: memref<4xf32>) -> () {
#         func.return
#     }
#     """
#
#     result = run_conversion(test_input)
#     print(result)
