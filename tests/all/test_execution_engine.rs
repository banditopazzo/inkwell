use std::cell::RefCell;
use std::rc::Rc;

use inkwell::context::Context;
use inkwell::execution_engine::{DataLayoutError, FunctionLookupError, RemoveModuleError};
use inkwell::memory_manager::McjitMemoryManager;
use inkwell::module::Linkage;
use inkwell::targets::{CodeModel, InitializationConfig, Target, TargetMachine};
use inkwell::{AddressSpace, IntPredicate, OptimizationLevel};

#[cfg(target_os = "windows")]
use windows::Win32::System::{
    Memory::{
        MEM_COMMIT, MEM_RELEASE, MEM_RESERVE, PAGE_EXECUTE_READ, PAGE_READWRITE, VirtualAlloc, VirtualFree,
        VirtualProtect,
    },
    SystemInformation::{GetSystemInfo, SYSTEM_INFO},
};

type Thunk = unsafe extern "C" fn();

#[test]
fn test_get_function_address() {
    let context = Context::create();
    // let module = context.create_module("errors_abound");
    let builder = context.create_builder();
    let void_type = context.void_type();
    let fn_type = void_type.fn_type(&[], false);

    // FIXME: LLVM's global state is leaking, causing this to fail in `cargo test` but not `cargo test test_get_function_address`
    // nor (most of the time) with `cargo test -- --test-threads LARGE_NUM`
    // assert_eq!(module.create_jit_execution_engine(OptimizationLevel::None), Err("Unable to find target for this triple (no targets are registered)".into()));
    let module = context.create_module("errors_abound");

    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

    unsafe {
        assert_eq!(
            execution_engine.get_function::<Thunk>("errors").unwrap_err(),
            FunctionLookupError::FunctionNotFound
        );
    }

    let module = context.create_module("errors_abound");
    let fn_value = module.add_function("func", fn_type, None);
    let basic_block = context.append_basic_block(fn_value, "entry");

    builder.position_at_end(basic_block);
    builder.build_return(None).unwrap();

    let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

    unsafe {
        assert_eq!(
            execution_engine.get_function::<Thunk>("errors").unwrap_err(),
            FunctionLookupError::FunctionNotFound
        );

        assert!(execution_engine.get_function::<Thunk>("func").is_ok());
    }
}

#[test]
fn test_jit_execution_engine() {
    let context = Context::create();
    let module = context.create_module("main_module");
    let builder = context.create_builder();
    let i32_type = context.i32_type();
    #[cfg(feature = "typed-pointers")]
    let i8_ptr_ptr_type = context
        .i8_type()
        .ptr_type(AddressSpace::default())
        .ptr_type(AddressSpace::default());
    #[cfg(not(feature = "typed-pointers"))]
    let i8_ptr_ptr_type = context.ptr_type(AddressSpace::default());
    let one_i32 = i32_type.const_int(1, false);
    let three_i32 = i32_type.const_int(3, false);
    let fourtytwo_i32 = i32_type.const_int(42, false);
    let fn_type = i32_type.fn_type(&[i32_type.into(), i8_ptr_ptr_type.into()], false);
    let fn_value = module.add_function("main", fn_type, None);
    let main_argc = fn_value.get_first_param().unwrap().into_int_value();
    // let main_argv = fn_value.get_nth_param(1).unwrap();
    let check_argc = context.append_basic_block(fn_value, "check_argc");
    let check_arg3 = context.append_basic_block(fn_value, "check_arg3");
    let error1 = context.append_basic_block(fn_value, "error1");
    let success = context.append_basic_block(fn_value, "success");

    main_argc.set_name("argc");

    // If anything goes wrong, jump to returning 1
    builder.position_at_end(error1);
    builder.build_return(Some(&one_i32)).unwrap();

    // If successful, jump to returning 42
    builder.position_at_end(success);
    builder.build_return(Some(&fourtytwo_i32)).unwrap();

    // See if argc == 3
    builder.position_at_end(check_argc);

    let eq = IntPredicate::EQ;
    let argc_check = builder.build_int_compare(eq, main_argc, three_i32, "argc_cmp").unwrap();

    builder
        .build_conditional_branch(argc_check, check_arg3, error1)
        .unwrap();

    builder.position_at_end(check_arg3);
    builder.build_unconditional_branch(success).unwrap();

    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let execution_engine = module
        .create_jit_execution_engine(OptimizationLevel::None)
        .expect("Could not create Execution Engine");

    let main = execution_engine
        .get_function_value("main")
        .expect("Could not find main in ExecutionEngine");

    let ret = unsafe { execution_engine.run_function_as_main(main, &["input", "bar"]) };

    assert_eq!(ret, 1, "unexpected main return code: {ret}");

    let ret = unsafe { execution_engine.run_function_as_main(main, &["input", "bar", "baz"]) };

    assert_eq!(ret, 42, "unexpected main return code: {ret}");
}

// #[test]
// fn test_execution_engine_empty_module() {
//     let context = Context::create();
//     let module = context.create_module("fooo");
//     let builder = context.create_builder();

//     let ee = module.create_jit_execution_engine(OptimizationLevel::None); // Segfault?
// }

#[test]
fn test_execution_engine() {
    let context = Context::create();
    let module = context.create_module("main_module");

    assert!(module.create_execution_engine().is_ok());
}

#[test]
fn test_interpreter_execution_engine() {
    let context = Context::create();
    let module = context.create_module("main_module");

    assert!(module.create_interpreter_execution_engine().is_ok());
}

#[test]

fn test_mcjit_execution_engine_with_memory_manager() {
    let mmgr = MockMemoryManager::new();
    let mmgr_for_test = mmgr.clone();

    {
        let context = Context::create();
        let module = context.create_module("main_module");
        let builder = context.create_builder();

        // Define @llvm.experimental.stackmap
        let fn_type = context
            .void_type()
            .fn_type(&[context.i64_type().into(), context.i32_type().into()], true);
        let stackmap_func = module.add_function("llvm.experimental.stackmap", fn_type, Some(Linkage::External));

        // Set up the function
        //
        // `@llvm.experimental.stackmap` a call is present, LLVM will emit a separate stackmap section,
        // causing the `allocate_data_section` callback to be invoked an additional
        // time to handle the stackmap data. Specifically:
        // ```
        // f64 test_fn() {
        // entry:
        //   call void @llvm.experimental.stackmap(i64 12345, i32 0)
        //   ret f64 64.0
        // }
        // ```
        let double = context.f64_type();
        let sig = double.fn_type(&[], false);
        let f = module.add_function("test_fn", sig, None);
        let b = context.append_basic_block(f, "entry");
        builder.position_at_end(b);

        // Create a call to the stackmap intrinsic
        // Stack maps are used by the garbage collector to find roots on the stack
        // See: https://llvm.org/docs/StackMaps.html#intrinsics
        builder
            .build_call(
                stackmap_func,
                &[
                    context.i64_type().const_int(12345, false).into(),
                    context.i32_type().const_int(0, false).into(),
                ],
                "call_stackmap",
            )
            .unwrap();

        // Insert a return statement
        let ret = double.const_float(64.0);
        builder.build_return(Some(&ret)).unwrap();

        module.verify().unwrap();

        let ee = module
            .create_mcjit_execution_engine_with_memory_manager(
                mmgr,
                OptimizationLevel::None,
                CodeModel::Default,
                false,
                false,
            )
            .unwrap();

        unsafe {
            let test_fn = ee.get_function::<unsafe extern "C" fn() -> f64>("test_fn").unwrap();
            let return_value = test_fn.call();
            assert_eq!(return_value, 64.0);
        }
    }
    // ee dropped here. Destroy should be called

    let data = mmgr_for_test.data.borrow();
    assert_eq!(1, data.code_alloc_calls);
    assert!(
        data.data_alloc_calls >= 2,
        "Expected at least 2 calls to allocate_data_section, but got {}. \
         We've observed that LLVM 5 typically calls it 3 times, while LLVM 18 often calls it only 2.",
        data.data_alloc_calls
    );
    assert_eq!(1, data.finalize_calls);
    assert_eq!(1, data.destroy_calls);
}

#[test]
fn test_add_remove_module() {
    let context = Context::create();
    let ee = context
        .create_module("test")
        .create_jit_execution_engine(OptimizationLevel::default())
        .unwrap();

    // The bootstrap module's handle is handed out once
    let test = ee.take_initial_loaded_module().unwrap();
    assert!(ee.take_initial_loaded_module().is_none());

    // Module names need not be unique
    let module2 = ee.add_module(context.create_module("mod2")).unwrap();
    let module3 = ee.add_module(context.create_module("mod2")).unwrap();

    // A handle from another engine is rejected; that engine keeps its module
    let ee2 = context
        .create_module("other")
        .create_jit_execution_engine(OptimizationLevel::default())
        .unwrap();
    let other = ee2.add_module(context.create_module("other2")).unwrap();

    let other = match ee.remove_module(other) {
        Err(RemoveModuleError::ModuleNotOwned(other)) => other,
        result => panic!("expected ModuleNotOwned, got {result:?}"),
    };
    // ... and the handle comes back, so the module is not lost
    assert_eq!(ee2.remove_module(other).unwrap().get_name().to_str(), Ok("other2"));

    // Removed modules come back as plain modules we own again, the bootstrap one included
    let module2 = ee.remove_module(module2).unwrap();
    let module3 = ee.remove_module(module3).unwrap();
    let test = ee.remove_module(test).unwrap();

    assert_eq!(module2.get_name().to_str(), Ok("mod2"));
    assert_eq!(module3.get_name().to_str(), Ok("mod2"));
    assert_eq!(test.get_name().to_str(), Ok("test"));
    assert_ne!(module2.as_mut_ptr(), module3.as_mut_ptr());
}

#[test]
fn test_static_constructors_without_modules() {
    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let context = Context::create();
    let builder = context.create_builder();
    let i64_type = context.i64_type();

    let ee = context
        .create_module("boot")
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    let boot = ee.take_initial_loaded_module().unwrap();

    // An engine that owns nothing and never compiled anything must not crash here
    // (MCJIT::finalizeObject would dereference a RuntimeDyld that does not exist yet).
    drop(ee.remove_module(boot).unwrap());
    ee.run_static_constructors();
    ee.run_static_destructors();
    assert_eq!(
        ee.get_function_address("anything").unwrap_err(),
        FunctionLookupError::FunctionNotFound
    );

    // It keeps working with modules added afterwards ...
    let m = context.create_module("later");
    let f = m.add_function("seven", i64_type.fn_type(&[], false), None);
    builder.position_at_end(context.append_basic_block(f, "entry"));
    builder.build_return(Some(&i64_type.const_int(7, false))).unwrap();
    let later = ee.add_module(m).unwrap();
    ee.run_static_constructors();

    let seven = unsafe { ee.get_function::<unsafe extern "C" fn() -> i64>("seven").unwrap() };
    assert_eq!(unsafe { seven.call() }, 7);

    // ... and once something has been compiled, finalizing with no owned modules is fine.
    drop(ee.remove_module(later).unwrap());
    ee.run_static_constructors();
    ee.run_static_destructors();
}

#[test]
fn test_code_survives_module_removal() {
    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let context = Context::create();
    let builder = context.create_builder();
    let i64_type = context.i64_type();
    let fn_type = i64_type.fn_type(&[], false);

    let ee = context
        .create_module("main")
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();

    let extra = context.create_module("extra");
    let f = extra.add_function("seven", fn_type, None);
    builder.position_at_end(context.append_basic_block(f, "entry"));
    builder.build_return(Some(&i64_type.const_int(7, false))).unwrap();
    let extra = ee.add_module(extra).unwrap();

    let seven = unsafe { ee.get_function::<unsafe extern "C" fn() -> i64>("seven").unwrap() };
    assert_eq!(unsafe { seven.call() }, 7);

    // Taking the module back (and dropping it) does not free the generated code
    let extra = ee.remove_module(extra).unwrap();
    assert_eq!(extra.get_name().to_str(), Ok("extra"));
    drop(extra);

    assert_eq!(unsafe { seven.call() }, 7);

    let seven = unsafe { ee.get_function::<unsafe extern "C" fn() -> i64>("seven").unwrap() };
    assert_eq!(unsafe { seven.call() }, 7);

    // The bootstrap module can be taken back too
    let main = ee.remove_module(ee.take_initial_loaded_module().unwrap()).unwrap();
    assert_eq!(main.get_name().to_str(), Ok("main"));
}

// REVIEW: Global state pollution access tests cause this to pass when run individually
// but fail when multiple tests are run
// #[test]
// fn test_no_longer_segfaults() {
//     Target::initialize_amd_gpu(&InitializationConfig::default());

//     let context = Context::create();
//     let module = context.create_module("test");

//     assert_eq!(*module.create_jit_execution_engine(OptimizationLevel::default()).unwrap_err(), *CString::new("No available targets are compatible with this triple.").unwrap());

//     // REVIEW: Module is being cloned in err case... should test Module still works as expected...
// }

// #[test]
// fn test_get_function_value() {
//     let context = Context::create();
//     let builder = context.create_builder();
//     let module = context.create_module("errors_abound");
//     // let mut execution_engine = ExecutionEngine::create_jit_from_module(module, 0);
//     let mut (execution_engine, _) = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();
//     let void_type = context.void_type();
//     let fn_type = void_type.fn_type(&[], false);
//     let fn_value = module.add_function("func", fn_type, None);
//     let basic_block = context.append_basic_block(&fn_value, "entry");

//     builder.position_at_end(basic_block);
//     builder.build_return(None);

//     assert_eq!(execution_engine.get_function_value("errors"), Err(FunctionLookupError::JITNotEnabled));

//     // Regain ownership of module
//     let module = execution_engine.remove_module(&module).unwrap();

//     Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

//     let execution_engine = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

//     assert_eq!(execution_engine.get_function_value("errors"), Err(FunctionLookupError::FunctionNotFound));

//     assert!(execution_engine.get_function_value("func").is_ok());
// }

// No longer necessary with explicit lifetimes
// #[test]
// fn test_previous_double_free() {
//     Target::initialize_native(&InitializationConfig::default()).unwrap();

//     let context = Context::create();
//     let module = context.create_module("sum");
//     let _ee = module.create_jit_execution_engine(OptimizationLevel::None).unwrap();

//     drop(context);
//     drop(module);
// }

// #[test]
// fn test_previous_double_free2() {
//     Target::initialize_native(&InitializationConfig::default()).unwrap();

//     let _execution_engine = {
//         let context = Context::create();
//         let module = context.create_module("sum");

//         module.create_jit_execution_engine(OptimizationLevel::None).unwrap()
//     };
// }

/// A mock memory manager that allocates memory in fixed-size pages for testing.
#[derive(Debug, Clone)]
struct MockMemoryManager {
    data: Rc<RefCell<MockMemoryManagerData>>,
}

#[derive(Debug)]
struct MockMemoryManagerData {
    fixed_capacity_bytes: usize,
    fixed_page_size: usize,

    code_buff_ptr: std::ptr::NonNull<u8>,
    code_offset: usize,

    data_buff_ptr: std::ptr::NonNull<u8>,
    data_offset: usize,

    /// Count call to callbacks for testing
    code_alloc_calls: usize,
    data_alloc_calls: usize,
    finalize_calls: usize,
    destroy_calls: usize,
}

impl MockMemoryManager {
    pub fn new() -> Self {
        let capacity_bytes = 128 * 1024;
        #[cfg(unix)]
        let page_size = unsafe { libc::sysconf(libc::_SC_PAGESIZE) as usize };

        #[cfg(target_os = "windows")]
        let page_size = {
            let mut info = std::mem::MaybeUninit::<SYSTEM_INFO>::uninit();
            unsafe {
                GetSystemInfo(info.as_mut_ptr());
            }
            let info = unsafe { info.assume_init() };
            info.dwPageSize as usize
        };

        #[cfg(unix)]
        let code_buff_ptr = unsafe {
            std::ptr::NonNull::new_unchecked(libc::mmap(
                std::ptr::null_mut(),
                capacity_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8)
        };

        #[cfg(target_os = "windows")]
        let code_buff_ptr = unsafe {
            std::ptr::NonNull::new_unchecked(
                VirtualAlloc(None, capacity_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>(),
            )
        };

        #[cfg(unix)]
        let data_buff_ptr = unsafe {
            std::ptr::NonNull::new_unchecked(libc::mmap(
                std::ptr::null_mut(),
                capacity_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
                libc::MAP_PRIVATE | libc::MAP_ANONYMOUS,
                -1,
                0,
            ) as *mut u8)
        };

        #[cfg(target_os = "windows")]
        let data_buff_ptr = unsafe {
            std::ptr::NonNull::new_unchecked(
                VirtualAlloc(None, capacity_bytes, MEM_COMMIT | MEM_RESERVE, PAGE_READWRITE).cast::<u8>(),
            )
        };

        Self {
            data: Rc::new(RefCell::new(MockMemoryManagerData {
                fixed_capacity_bytes: capacity_bytes,
                fixed_page_size: page_size,

                code_buff_ptr,
                code_offset: 0,

                data_buff_ptr,
                data_offset: 0,

                code_alloc_calls: 0,
                data_alloc_calls: 0,
                finalize_calls: 0,
                destroy_calls: 0,
            })),
        }
    }
}

impl McjitMemoryManager for MockMemoryManager {
    fn allocate_code_section(
        &mut self,
        size: libc::size_t,
        _alignment: libc::c_uint,
        _section_id: libc::c_uint,
        _section_name: &str,
    ) -> *mut u8 {
        let mut data = self.data.borrow_mut();
        data.code_alloc_calls += 1;

        let alloc_size = size.div_ceil(data.fixed_page_size) * data.fixed_page_size;
        let ptr = unsafe { data.code_buff_ptr.as_ptr().add(data.code_offset) };
        data.code_offset += alloc_size;

        ptr
    }

    fn allocate_data_section(
        &mut self,
        size: libc::size_t,
        _alignment: libc::c_uint,
        _section_id: libc::c_uint,
        _section_name: &str,
        _is_read_only: bool,
    ) -> *mut u8 {
        let mut data = self.data.borrow_mut();

        data.data_alloc_calls += 1;

        let alloc_size = size.div_ceil(data.fixed_page_size) * data.fixed_page_size;
        let ptr = unsafe { data.data_buff_ptr.as_ptr().add(data.data_offset) };
        data.data_offset += alloc_size;

        ptr
    }

    fn finalize_memory(&mut self) -> Result<(), String> {
        let mut data = self.data.borrow_mut();

        data.finalize_calls += 1;

        #[cfg(unix)]
        unsafe {
            libc::mprotect(
                data.code_buff_ptr.as_ptr() as *mut libc::c_void,
                data.fixed_capacity_bytes,
                libc::PROT_READ | libc::PROT_EXEC,
            );
            libc::mprotect(
                data.data_buff_ptr.as_ptr() as *mut libc::c_void,
                data.fixed_capacity_bytes,
                libc::PROT_READ | libc::PROT_WRITE,
            );
        }

        #[cfg(windows)]
        unsafe {
            let mut old_protect = PAGE_READWRITE;
            VirtualProtect(
                data.code_buff_ptr.as_ptr() as *mut _,
                data.fixed_capacity_bytes,
                PAGE_EXECUTE_READ,
                &mut old_protect,
            )
            .expect("VirtualProtect failed");
            VirtualProtect(
                data.data_buff_ptr.as_ptr() as *mut _,
                data.fixed_capacity_bytes,
                PAGE_READWRITE,
                &mut old_protect,
            )
            .expect("VirtualProtect failed");
        }

        Ok(())
    }

    fn destroy(&mut self) {
        let mut data = self.data.borrow_mut();

        data.destroy_calls += 1;

        #[cfg(unix)]
        unsafe {
            libc::munmap(
                data.code_buff_ptr.as_ptr() as *mut libc::c_void,
                data.fixed_capacity_bytes,
            );
            libc::munmap(
                data.data_buff_ptr.as_ptr() as *mut libc::c_void,
                data.fixed_capacity_bytes,
            );
        }

        #[cfg(windows)]
        unsafe {
            VirtualFree(data.code_buff_ptr.as_ptr() as *mut _, 0, MEM_RELEASE).expect("Failed to free memory.");
            VirtualFree(data.data_buff_ptr.as_ptr() as *mut _, 0, MEM_RELEASE).expect("Failed to free memory.");
        }
    }
}

#[test]
fn test_raw_module_and_engine_constructors() {
    use inkwell::execution_engine::ExecutionEngine;
    use inkwell::llvm_sys::execution_engine::{LLVMAddModule, LLVMCreateJITCompilerForModule};

    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let context = Context::create();

    // An engine created around a raw module knows that module
    let module = context.create_module("raw");
    let raw = module.as_mut_ptr();
    std::mem::forget(module); // the engine owns it from here on
    let ee = unsafe { ExecutionEngine::create_jit_for_module(raw, OptimizationLevel::None) }.unwrap();
    let initial = ee.take_initial_loaded_module().unwrap();
    assert!(ee.take_initial_loaded_module().is_none());
    assert_eq!(ee.remove_module(initial).unwrap().get_name().to_str(), Ok("raw"));

    // So does an engine created through llvm-sys and wrapped afterwards ...
    let module = context.create_module("wrapped");
    let raw = module.as_mut_ptr();
    std::mem::forget(module);
    let mut raw_ee = std::ptr::null_mut();
    let mut err = std::ptr::null_mut();
    assert_eq!(
        unsafe { LLVMCreateJITCompilerForModule(&mut raw_ee, raw, 0, &mut err) },
        0
    );
    let ee = unsafe { ExecutionEngine::new(raw_ee, raw) };
    let wrapped = ee.remove_module(ee.take_initial_loaded_module().unwrap()).unwrap();
    assert_eq!(wrapped.get_name().to_str(), Ok("wrapped"));

    // ... and a module added through the raw API can be tracked and taken back
    let extra = context.create_module("extra");
    let raw_extra = extra.as_mut_ptr();
    std::mem::forget(extra);
    unsafe { LLVMAddModule(ee.as_mut_ptr(), raw_extra) };
    let extra = unsafe { ee.track_module(raw_extra) };
    assert_eq!(ee.remove_module(extra).unwrap().get_name().to_str(), Ok("extra"));

    // The engine owns nothing now and never compiled anything: still no crash
    ee.run_static_constructors();
    ee.run_static_destructors();

    // The default engine kind is observed, not assumed: MCJIT has a target machine, the
    // interpreter does not, and `get_function` goes by that
    let module = context.create_module("default");
    let raw = module.as_mut_ptr();
    std::mem::forget(module);
    let ee = unsafe { ExecutionEngine::create_for_module(raw) }.unwrap();
    let has_jit = Target::from_triple(&TargetMachine::get_default_triple())
        .unwrap()
        .has_jit();
    assert_eq!(ee.get_target_machine().is_some(), has_jit);
    assert_eq!(
        ee.get_function_address("nothing").unwrap_err(),
        if has_jit {
            FunctionLookupError::FunctionNotFound
        } else {
            FunctionLookupError::JITNotEnabled
        }
    );
    if let Some(target_machine) = ee.get_target_machine() {
        // MCJIT's own machine: same layout as the engine reports, borrowed, not disposed
        assert_eq!(target_machine.get_data_layout(), ee.get_data_layout());
        assert!(target_machine.get_target().has_jit());
    }

    let module = context.create_module("interp");
    let raw = module.as_mut_ptr();
    std::mem::forget(module);
    let interpreter = unsafe { ExecutionEngine::create_interpreter_for_module(raw) }.unwrap();
    assert!(interpreter.get_target_machine().is_none());
    assert_eq!(
        interpreter.get_function_address("nothing").unwrap_err(),
        FunctionLookupError::JITNotEnabled
    );
}

#[test]
fn test_data_layout_must_match_the_engine() {
    use inkwell::module::Module;

    /// `name() -> i64`, the offset of the `i64` in `{ i8, i64 }`, in a module with `data_layout`.
    fn with_layout<'ctx>(context: &'ctx Context, name: &str, data_layout: &str) -> Module<'ctx> {
        let module = context.create_module(name);
        module.set_data_layout_str(data_layout);
        let i64_type = context.i64_type();
        let pair = context.struct_type(&[context.i8_type().into(), i64_type.into()], false);
        let function = module.add_function(name, i64_type.fn_type(&[], false), None);
        let builder = context.create_builder();
        builder.position_at_end(context.append_basic_block(function, "entry"));
        let indexes = [i64_type.const_zero(), context.i32_type().const_int(1, false)];
        #[cfg(feature = "typed-pointers")]
        let field = unsafe { pair.ptr_type(AddressSpace::default()).const_null().const_gep(&indexes) };
        #[cfg(not(feature = "typed-pointers"))]
        let field = unsafe {
            context
                .ptr_type(AddressSpace::default())
                .const_null()
                .const_gep(pair, &indexes)
        };
        builder.build_return(Some(&field.const_to_int(i64_type))).unwrap();
        module
    }
    const FOREIGN: &str = "e-m:e-i64:32-n8:16:32:64-S128";

    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let context = Context::create();
    let ee = context
        .create_module("main")
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    let engine_layout = ee
        .get_data_layout()
        .get_string_representation()
        .to_string_lossy()
        .into_owned();

    // A module with another layout is refused and given back, instead of ending the process when
    // MCJIT would compile it
    let refused = ee.add_module(with_layout(&context, "refused", FOREIGN)).unwrap_err();
    assert_eq!(refused.module.get_name().to_str(), Ok("refused"));
    assert_eq!(
        refused.error,
        DataLayoutError::LayoutMismatch {
            module_data_layout: FOREIGN.to_owned(),
            engine_data_layout: engine_layout.clone(),
        }
    );
    assert!(refused.to_string().contains(FOREIGN));

    // The engine's own layout, and no layout at all (MCJIT stamps its own), are accepted
    let same = ee.add_module(with_layout(&context, "same", &engine_layout)).unwrap();
    let none = context.create_module("none");
    let i64_type = context.i64_type();
    let function = none.add_function("none", i64_type.fn_type(&[], false), None);
    let builder = context.create_builder();
    builder.position_at_end(context.append_basic_block(function, "entry"));
    builder.build_return(Some(&i64_type.const_int(0, false))).unwrap();
    let none = ee.add_module(none).unwrap();

    let offset = unsafe { ee.get_function::<unsafe extern "C" fn() -> i64>("same").unwrap() };
    assert_eq!(unsafe { offset.call() }, 8);
    let none = ee.remove_module(none).unwrap();
    assert_eq!(
        none.get_data_layout().get_string_representation().to_str(),
        Ok(engine_layout.as_str())
    );
    drop(ee.remove_module(same).unwrap());

    // An engine created around a module with another layout is refused too
    let error = with_layout(&context, "first", FOREIGN)
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap_err();
    assert!(error.to_string().contains(FOREIGN), "{error}");

    // The interpreter runs under the host's layout too, which MCJIT's is: another one is refused,
    // first or added, as the interpreter's memory is shared with native code
    let error = with_layout(&context, "first", FOREIGN)
        .create_interpreter_execution_engine()
        .unwrap_err();
    assert!(error.to_string().contains(FOREIGN), "{error}");
    let interpreter = with_layout(&context, "first", &engine_layout)
        .create_interpreter_execution_engine()
        .unwrap();
    assert_eq!(
        interpreter.get_data_layout().get_string_representation().to_str(),
        Ok(engine_layout.as_str())
    );
    let refused = interpreter
        .add_module(with_layout(&context, "foreign", FOREIGN))
        .unwrap_err();
    assert_eq!(refused.module.get_name().to_str(), Ok("foreign"));
    let same = interpreter
        .add_module(with_layout(&context, "same", &engine_layout))
        .unwrap();
    let function = interpreter.get_function_value("same").unwrap();
    assert_eq!(unsafe { interpreter.run_function(function, &[]) }.as_int(false), 8);
    drop(interpreter.remove_module(same).unwrap());

    // Modules without a layout, first or added, are given the host's
    let interpreter = context
        .create_module("main")
        .create_interpreter_execution_engine()
        .unwrap();
    assert_eq!(
        interpreter.get_data_layout().get_string_representation().to_str(),
        Ok(engine_layout.as_str())
    );
    let explicit = interpreter
        .add_module(with_layout(&context, "explicit", &engine_layout))
        .unwrap();
    let empty = interpreter.add_module(context.create_module("empty")).unwrap();
    let empty = interpreter.remove_module(empty).unwrap();
    assert_eq!(
        empty.get_data_layout().get_string_representation().to_str(),
        Ok(engine_layout.as_str())
    );
    drop(interpreter.remove_module(explicit).unwrap());
}

/// The interpreter stores every pointer as a host pointer, into a slot of the layout's size for
/// its address space. x86-64's layout has 32-bit address spaces 270 and 271: a module using them
/// is refused by the interpreter, before its constructor stores the module's globals, and
/// accepted by MCJIT.
#[cfg(target_arch = "x86_64")]
#[test]
fn test_interpreter_refuses_address_spaces_of_another_pointer_size() {
    use inkwell::module::Module;

    const POINTER_SIZE: DataLayoutError = DataLayoutError::PointerSize {
        address_space: 270,
        size: 4,
        host_size: 8,
    };

    /// A global `ptr addrspace(270)`, which `emitGlobals` would store 8 bytes into.
    fn global(context: &Context) -> Module<'_> {
        let module = context.create_module("global");
        #[cfg(feature = "typed-pointers")]
        let pointer_type = context.i8_type().ptr_type(AddressSpace::from(270));
        #[cfg(not(feature = "typed-pointers"))]
        let pointer_type = context.ptr_type(AddressSpace::from(270));
        let pointer = module.add_global(pointer_type, None, "pointer");
        pointer.set_initializer(&pointer_type.const_null());
        module
    }

    /// The address space only in an `alloca` of a function.
    fn alloca(context: &Context) -> Module<'_> {
        let module = context.create_module("alloca");
        let function = module.add_function("alloca", context.void_type().fn_type(&[], false), None);
        let builder = context.create_builder();
        builder.position_at_end(context.append_basic_block(function, "entry"));
        #[cfg(feature = "typed-pointers")]
        let pointer_type = context.i8_type().ptr_type(AddressSpace::from(270));
        #[cfg(not(feature = "typed-pointers"))]
        let pointer_type = context.ptr_type(AddressSpace::from(270));
        builder.build_alloca(pointer_type, "slot").unwrap();
        builder.build_return(None).unwrap();
        module
    }

    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let context = Context::create();
    let error = global(&context).create_interpreter_execution_engine().unwrap_err();
    assert_eq!(error.to_string(), POINTER_SIZE.to_string());
    // The default kind may become the interpreter, so it checks too
    let error = global(&context).create_execution_engine().unwrap_err();
    assert_eq!(error.to_string(), POINTER_SIZE.to_string());

    let interpreter = context
        .create_module("main")
        .create_interpreter_execution_engine()
        .unwrap();
    let refused = interpreter.add_module(global(&context)).unwrap_err();
    assert_eq!(refused.error, POINTER_SIZE);
    let refused = interpreter.add_module(alloca(&context)).unwrap_err();
    assert_eq!(refused.error, POINTER_SIZE);

    let ee = global(&context)
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap();
    ee.add_module(alloca(&context)).unwrap();
}

/// MCJIT builds its target machine from the module's target triple, and its code is called from
/// this process: a module for another target is refused, whatever its layout.
#[cfg(all(target_arch = "x86_64", target_os = "linux"))]
#[test]
fn test_jit_refuses_foreign_targets() {
    use inkwell::targets::TargetTriple;

    Target::initialize_native(&InitializationConfig::default()).expect("Failed to initialize native target");

    let context = Context::create();
    // Another architecture of the same backend, and the host's architecture with another ABI
    for triple in ["i386-pc-linux-gnu", "x86_64-pc-windows-msvc"] {
        let module = context.create_module(triple);
        module.set_triple(&TargetTriple::create(triple));
        let error = module.create_jit_execution_engine(OptimizationLevel::None).unwrap_err();
        assert!(error.to_string().contains(triple), "{error}");
    }

    // With the host's layout set, the target is still another
    let module = context.create_module("i386");
    module.set_triple(&TargetTriple::create("i386-pc-linux-gnu"));
    let host_layout = context
        .create_module("host")
        .create_jit_execution_engine(OptimizationLevel::None)
        .unwrap()
        .get_data_layout()
        .get_string_representation()
        .to_string_lossy()
        .into_owned();
    module.set_data_layout_str(&host_layout);
    let error = module.create_jit_execution_engine(OptimizationLevel::None).unwrap_err();
    assert!(error.to_string().contains("i386-pc-linux-gnu"), "{error}");

    // The host's own triple is accepted
    let module = context.create_module("host");
    module.set_triple(&TargetMachine::get_default_triple());
    drop(module.create_jit_execution_engine(OptimizationLevel::None).unwrap());
}
