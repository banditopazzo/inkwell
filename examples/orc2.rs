//! ORC2 LLJIT basic usage example.
//!
//! Builds a `sum(a, b, c) -> a + b + c` function using inkwell's IR builder,
//! compiles it through ORC2's LLJIT engine, and calls the result.

use inkwell::orc2::{LLJit, ThreadSafeContext};
use inkwell::targets::{InitializationConfig, Target};

use std::error::Error;

type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

fn main() -> Result<(), Box<dyn Error>> {
    Target::initialize_native(&InitializationConfig::default())
        .map_err(|e| format!("Failed to initialize native target: {e}"))?;

    let lljit = LLJit::create()?;

    let tsc = ThreadSafeContext::create();
    let ctx = tsc.context();
    let module = ctx.create_module("sum");
    let builder = ctx.create_builder();

    let i64_type = ctx.i64_type();
    let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
    let function = module.add_function("sum", fn_type, None);
    let basic_block = ctx.append_basic_block(function, "entry");
    builder.position_at_end(basic_block);

    let x = function.get_nth_param(0).unwrap().into_int_value();
    let y = function.get_nth_param(1).unwrap().into_int_value();
    let z = function.get_nth_param(2).unwrap().into_int_value();
    let sum = builder.build_int_add(x, y, "sum").unwrap();
    let sum = builder.build_int_add(sum, z, "sum").unwrap();
    builder.build_return(Some(&sum)).unwrap();

    let tsm = tsc
        .create_thread_safe_module(module)
        .expect("Module was not created from this ThreadSafeContext");
    lljit.add_module(&lljit.main_jit_dylib(), tsm)?;

    let sum = unsafe { lljit.get_function::<SumFunc>("sum")? };

    let x = 1u64;
    let y = 2u64;
    let z = 3u64;

    unsafe {
        println!("{} + {} + {} = {}", x, y, z, sum.call(x, y, z));
        assert_eq!(sum.call(x, y, z), x + y + z);
    }

    Ok(())
}
