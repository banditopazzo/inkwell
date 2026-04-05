//! ORC2 JIT compilation support.
//!
//! # Why LLVM 18+?
//!
//! The `llvm-sys` crate only exposes the `orc2` module starting from version 181 (LLVM 18.1).
//! Earlier versions only had ORC v1 bindings (`LLVMOrcJITStackRef`), a different API.
//!
//! # Omitted symbols
//!
//! `LLVMOrcMaterializationResponsibilityAddDependencies` and
//! `LLVMOrcMaterializationResponsibilityAddDependenciesForAll` were removed in LLVM 19
//! and are intentionally not wrapped.
//!
//! # Example
//!
//! ```ignore
//! use inkwell::orc2::{ThreadSafeContext, LLJit};
//! use inkwell::targets::{InitializationConfig, Target};
//!
//! Target::initialize_native(&InitializationConfig::default()).unwrap();
//!
//! let tsc = ThreadSafeContext::create();
//! let ctx = tsc.context();
//! let module = ctx.create_module("example");
//! let builder = ctx.create_builder();
//! // ... build IR ...
//! let tsm = tsc.create_thread_safe_module(module).unwrap().unwrap();
//!
//! let lljit = LLJit::create()?;
//! lljit.add_module(&lljit.main_jit_dylib(), tsm)?;
//! # Ok::<(), Box<dyn std::error::Error>>(())
//! ```

use llvm_sys::error::LLVMGetErrorMessage;
use llvm_sys::orc2::ee::*;
use llvm_sys::orc2::lljit::*;
use llvm_sys::orc2::*;
use llvm_sys::prelude::{LLVMContextRef, LLVMMemoryBufferRef, LLVMModuleRef};

use crate::context::ContextRef;
use crate::execution_engine::UnsafeFunctionPointer;
use crate::memory_buffer::MemoryBuffer;
use crate::module::Module;
use crate::support::{LLVMString, to_c_str};

use std::ffi::CStr;
use std::fmt::{self, Debug, Formatter};
use std::marker::PhantomData;
use std::mem::{size_of, transmute_copy};

/// Converts an `LLVMErrorRef` into a `Result`.
fn into_result(err: llvm_sys::error::LLVMErrorRef) -> Result<(), LLVMString> {
    if err.is_null() {
        Ok(())
    } else {
        let msg = unsafe { LLVMGetErrorMessage(err) };
        unsafe { Err(LLVMString::new(msg)) }
    }
}

/// An ORC2 thread-safe context.
///
/// Wraps an `LLVMContext` with reference counting and a mutex for concurrent
/// access. Use [`context()`](Self::context) to get a [`ContextRef`] for building
/// IR, and [`create_thread_safe_module()`](Self::create_thread_safe_module) to
/// wrap finished modules for JIT compilation.
///
/// The lifetime parameter ties all [`ContextRef`]s and [`Module`]s created from
/// this context back to it, preventing mismatched module/context pairs at
/// compile time.
///
/// # Example
///
/// ```ignore
/// use inkwell::orc2::ThreadSafeContext;
///
/// let tsc = ThreadSafeContext::create();
/// let ctx = tsc.context();
/// let module = ctx.create_module("my_mod");
/// // ... build IR ...
/// let tsm = tsc.create_thread_safe_module(module).unwrap().unwrap();
/// ```
#[derive(Debug)]
pub struct ThreadSafeContext {
    raw: LLVMOrcThreadSafeContextRef,
    llvm_ctx: LLVMContextRef,
}

impl ThreadSafeContext {
    /// Creates a new thread-safe context.
    #[llvm_versions(..21)]
    pub fn create() -> Self {
        // LLVMOrcCreateNewThreadSafeContext creates a TSC with a fresh internal
        // context. LLVMOrcThreadSafeContextGetContext extracts the raw pointer.
        unsafe {
            let raw = LLVMOrcCreateNewThreadSafeContext();
            assert!(!raw.is_null());
            let llvm_ctx = LLVMOrcThreadSafeContextGetContext(raw);
            assert!(!llvm_ctx.is_null());
            ThreadSafeContext { raw, llvm_ctx }
        }
    }

    /// Creates a new thread-safe context.
    #[llvm_versions(21..)]
    pub fn create() -> Self {
        // LLVM 21 removed GetContext. We create a plain LLVMContext first, then
        // wrap it via LLVMOrcCreateNewThreadSafeContextFromLLVMContext.
        // FromLLVMContext transfers ownership but the raw pointer remains valid --
        // the TSC keeps the same allocation alive via internal refcounting.
        unsafe {
            let llvm_ctx = llvm_sys::core::LLVMContextCreate();
            assert!(!llvm_ctx.is_null());
            let raw = LLVMOrcCreateNewThreadSafeContextFromLLVMContext(llvm_ctx);
            assert!(!raw.is_null());
            ThreadSafeContext { raw, llvm_ctx }
        }
    }

    /// Returns a [`ContextRef`] for building IR.
    ///
    /// Use this to create modules, types, builders, and everything else that
    /// requires an LLVM context. Modules created from this context can be wrapped
    /// via [`create_thread_safe_module()`](Self::create_thread_safe_module).
    pub fn context(&self) -> ContextRef<'_> {
        unsafe { ContextRef::new(self.llvm_ctx) }
    }

    /// Converts a [`Module`] into a [`ThreadSafeModule`], consuming it.
    ///
    /// ORC2 takes ownership of the module. The module must have been created from
    /// the context returned by [`context()`](Self::context).
    ///
    /// # Errors
    ///
    /// Returns the module back inside an `Err` if it was created from a different
    /// context. LLVM performs no check for this, so we validate at runtime by
    /// comparing context pointers.
    pub fn create_thread_safe_module<'m>(&self, module: Module<'m>) -> Result<ThreadSafeModule, Module<'m>> {
        let module_ctx = module.get_context().raw();
        if module_ctx != self.llvm_ctx {
            return Err(module);
        }
        let module_ref = module.as_mut_ptr();
        // Suppress Module's Drop to prevent double-free. ORC2 takes ownership.
        std::mem::forget(module);
        // SAFETY: We just verified the module belongs to this context.
        // Destructor suppressed above.
        Ok(unsafe { self.wrap_module(module_ref) })
    }

    /// Wraps a raw `LLVMModuleRef` into a [`ThreadSafeModule`].
    ///
    /// # Safety
    ///
    /// - The module must have been created from the same `LLVMContext` that this
    ///   `ThreadSafeContext` wraps. Passing a module from a different context is
    ///   undefined behavior (silent data corruption, data races, use-after-free).
    ///   LLVM performs no runtime check for this.
    /// - The caller must have already suppressed the module wrapper's destructor
    ///   (e.g. via `std::mem::forget`) to prevent double-free, since ORC2 takes
    ///   ownership of the module.
    pub unsafe fn wrap_module(&self, module: LLVMModuleRef) -> ThreadSafeModule {
        unsafe {
            let raw = LLVMOrcCreateNewThreadSafeModule(module, self.raw);
            assert!(!raw.is_null());
            ThreadSafeModule { raw }
        }
    }

    /// Returns the raw `LLVMOrcThreadSafeContextRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcThreadSafeContextRef {
        self.raw
    }
}

impl Drop for ThreadSafeContext {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeThreadSafeContext(self.raw);
        }
    }
}

/// An LLVM module wrapped for thread-safe access by ORC2.
///
/// Created inside a [`with_thread_safe_context()`] closure. Owns the underlying
/// module and holds a reference-counted handle to the context. Once created,
/// the module can only be accessed through [`with_module_do()`](Self::with_module_do)
/// or consumed by [`LLJit::add_module()`].
#[derive(PartialEq, Eq)]
pub struct ThreadSafeModule {
    raw: LLVMOrcThreadSafeModuleRef,
}

impl ThreadSafeModule {
    /// Runs a callback with access to the module inside, under the context lock.
    ///
    /// This is the only way to inspect or modify the module after it has been
    /// wrapped. The callback receives a raw `LLVMModuleRef` which is valid only
    /// for the duration of the callback.
    ///
    /// # Safety
    ///
    /// The callback must not store the `LLVMModuleRef` beyond its own scope.
    pub unsafe fn with_module_do(
        &self,
        f: LLVMOrcGenericIRModuleOperationFunction,
        ctx: *mut libc::c_void,
    ) -> Result<(), LLVMString> {
        unsafe {
            let err = LLVMOrcThreadSafeModuleWithModuleDo(self.raw, f, ctx);
            into_result(err)
        }
    }

    /// Consumes this wrapper and returns the raw pointer without disposing.
    /// Used when transferring ownership to LLJIT.
    fn into_raw(self) -> LLVMOrcThreadSafeModuleRef {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }

    /// Returns the raw `LLVMOrcThreadSafeModuleRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference or use it after
    /// this wrapper is dropped or consumed by the JIT.
    pub unsafe fn as_raw(&self) -> LLVMOrcThreadSafeModuleRef {
        self.raw
    }
}

impl Debug for ThreadSafeModule {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("ThreadSafeModule").finish_non_exhaustive()
    }
}

impl Drop for ThreadSafeModule {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeThreadSafeModule(self.raw);
        }
    }
}

/// A non-owning reference to an ORC2 execution session.
///
/// The execution session is the top-level coordinator for all JIT state. It is
/// owned by the [`LLJit`] instance and must not outlive it.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ExecutionSession<'jit> {
    raw: LLVMOrcExecutionSessionRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl<'jit> ExecutionSession<'jit> {
    /// Interns a symbol name into the execution session's string pool.
    ///
    /// Returns a `SymbolStringPoolEntry` that can be used with ORC2 APIs
    /// that require interned names. The returned entry is reference-counted.
    pub fn intern(&self, name: &str) -> SymbolStringPoolEntry {
        let c_name = to_c_str(name);
        let raw = unsafe { LLVMOrcExecutionSessionIntern(self.raw, c_name.as_ptr()) };
        SymbolStringPoolEntry { raw }
    }

    /// Sets the error reporter callback for asynchronous errors.
    ///
    /// Errors that occur during materialization (compilation) are reported
    /// through this callback rather than returned from API calls.
    ///
    /// # Safety
    ///
    /// The callback and context pointer must remain valid for the lifetime of
    /// the execution session.
    pub unsafe fn set_error_reporter(&self, report_error: LLVMOrcErrorReporterFunction, ctx: *mut libc::c_void) {
        unsafe {
            LLVMOrcExecutionSessionSetErrorReporter(self.raw, report_error, ctx);
        }
    }

    /// Returns the symbol string pool for this execution session.
    pub fn get_symbol_string_pool(&self) -> SymbolStringPool<'jit> {
        let raw = unsafe { LLVMOrcExecutionSessionGetSymbolStringPool(self.raw) };
        SymbolStringPool {
            raw,
            _marker: PhantomData,
        }
    }

    /// Creates a bare JITDylib with the given name.
    ///
    /// A bare JITDylib has no link order set up. Use this for custom
    /// configurations.
    pub fn create_bare_jit_dylib(&self, name: &str) -> JitDylib<'jit> {
        let c_name = to_c_str(name);
        let raw = unsafe { LLVMOrcExecutionSessionCreateBareJITDylib(self.raw, c_name.as_ptr()) };
        assert!(!raw.is_null());
        JitDylib {
            raw,
            _marker: PhantomData,
        }
    }

    /// Creates a JITDylib with the given name and default link order.
    ///
    /// # Errors
    ///
    /// Returns an error if a JITDylib with this name already exists.
    pub fn create_jit_dylib(&self, name: &str) -> Result<JitDylib<'jit>, LLVMString> {
        let c_name = to_c_str(name);
        let mut jd = std::ptr::null_mut();
        let err = unsafe { LLVMOrcExecutionSessionCreateJITDylib(self.raw, &mut jd, c_name.as_ptr()) };
        into_result(err)?;
        assert!(!jd.is_null());
        Ok(JitDylib {
            raw: jd,
            _marker: PhantomData,
        })
    }

    /// Looks up a JITDylib by name. Returns `None` if not found.
    pub fn get_jit_dylib_by_name(&self, name: &str) -> Option<JitDylib<'jit>> {
        let c_name = to_c_str(name);
        let raw = unsafe { LLVMOrcExecutionSessionGetJITDylibByName(self.raw, c_name.as_ptr()) };
        if raw.is_null() {
            None
        } else {
            Some(JitDylib {
                raw,
                _marker: PhantomData,
            })
        }
    }

    /// Returns the raw `LLVMOrcExecutionSessionRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcExecutionSessionRef {
        self.raw
    }

    /// Performs an asynchronous symbol lookup.
    ///
    /// When the lookup completes (or fails), `handle_result` is called with the
    /// results. This is the non-blocking alternative to [`LLJit::lookup()`].
    ///
    /// # Safety
    ///
    /// All pointer parameters must be valid. The `handle_result` callback and
    /// `ctx` must remain valid until the callback is invoked.
    pub unsafe fn lookup(
        &self,
        kind: LLVMOrcLookupKind,
        search_order: LLVMOrcCJITDylibSearchOrder,
        search_order_size: usize,
        symbols: LLVMOrcCLookupSet,
        symbols_size: usize,
        handle_result: LLVMOrcExecutionSessionLookupHandleResultFunction,
        ctx: *mut libc::c_void,
    ) {
        unsafe {
            LLVMOrcExecutionSessionLookup(
                self.raw,
                kind,
                search_order,
                search_order_size,
                symbols,
                symbols_size,
                handle_result,
                ctx,
            );
        }
    }
}

/// A non-owning reference to an execution session's symbol string pool.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct SymbolStringPool<'jit> {
    raw: LLVMOrcSymbolStringPoolRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl SymbolStringPool<'_> {
    /// Clears dead entries from the pool, freeing unused memory.
    pub fn clear_dead_entries(&self) {
        unsafe {
            LLVMOrcSymbolStringPoolClearDeadEntries(self.raw);
        }
    }
}

/// An interned symbol name from the execution session's string pool.
///
/// Reference-counted. Cloning retains, dropping releases.
pub struct SymbolStringPoolEntry {
    raw: LLVMOrcSymbolStringPoolEntryRef,
}

impl SymbolStringPoolEntry {
    /// Returns the symbol name as a string slice.
    pub fn as_str(&self) -> &str {
        unsafe {
            let ptr = LLVMOrcSymbolStringPoolEntryStr(self.raw);
            CStr::from_ptr(ptr).to_str().expect("Symbol name is not valid UTF-8")
        }
    }

    /// Returns the raw `LLVMOrcSymbolStringPoolEntryRef`.
    pub fn as_raw(&self) -> LLVMOrcSymbolStringPoolEntryRef {
        self.raw
    }
}

impl Clone for SymbolStringPoolEntry {
    fn clone(&self) -> Self {
        unsafe {
            LLVMOrcRetainSymbolStringPoolEntry(self.raw);
        }
        SymbolStringPoolEntry { raw: self.raw }
    }
}

impl Drop for SymbolStringPoolEntry {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcReleaseSymbolStringPoolEntry(self.raw);
        }
    }
}

impl Debug for SymbolStringPoolEntry {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("SymbolStringPoolEntry").field(&self.as_str()).finish()
    }
}

/// A reference to a JIT dynamic library within an ORC2 execution session.
///
/// JITDylibs are virtual dynamic libraries that provide symbol scoping.
/// Non-owning -- the JITDylib is owned by the execution session.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct JitDylib<'jit> {
    raw: LLVMOrcJITDylibRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl<'jit> JitDylib<'jit> {
    /// Creates a new resource tracker for this JITDylib.
    pub fn create_resource_tracker(&self) -> ResourceTracker<'jit> {
        let raw = unsafe { LLVMOrcJITDylibCreateResourceTracker(self.raw) };
        assert!(!raw.is_null());
        ResourceTracker {
            raw,
            _marker: PhantomData,
        }
    }

    /// Returns the default resource tracker for this JITDylib.
    pub fn get_default_resource_tracker(&self) -> ResourceTracker<'jit> {
        let raw = unsafe { LLVMOrcJITDylibGetDefaultResourceTracker(self.raw) };
        assert!(!raw.is_null());
        ResourceTracker {
            raw,
            _marker: PhantomData,
        }
    }

    /// Defines a materialization unit in this JITDylib.
    ///
    /// Consumes the `MaterializationUnit`.
    ///
    /// # Errors
    ///
    /// Returns an error if duplicate symbols are defined.
    pub fn define(&self, mu: MaterializationUnit) -> Result<(), LLVMString> {
        let err = unsafe { LLVMOrcJITDylibDefine(self.raw, mu.into_raw()) };
        into_result(err)
    }

    /// Removes all definitions from this JITDylib.
    ///
    /// Useful for REPL-style redefinition.
    ///
    /// # Errors
    ///
    /// Returns an error if clearing fails.
    pub fn clear(&self) -> Result<(), LLVMString> {
        let err = unsafe { LLVMOrcJITDylibClear(self.raw) };
        into_result(err)
    }

    /// Adds a definition generator to this JITDylib.
    ///
    /// Generators are called when a symbol lookup fails, allowing on-demand
    /// symbol introduction (e.g., from the host process).
    ///
    /// Consumes the `DefinitionGenerator`.
    pub fn add_generator(&self, generator: DefinitionGenerator) {
        unsafe {
            LLVMOrcJITDylibAddGenerator(self.raw, generator.into_raw());
        }
    }

    /// Returns the raw `LLVMOrcJITDylibRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcJITDylibRef {
        self.raw
    }
}

/// Tracks resources (symbols, allocations) added to a JITDylib.
///
/// Enables code removal: call [`remove()`](Self::remove) to remove all tracked
/// symbols and resources.
#[derive(Debug, PartialEq, Eq)]
pub struct ResourceTracker<'jit> {
    raw: LLVMOrcResourceTrackerRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl<'jit> ResourceTracker<'jit> {
    /// Removes all resources tracked by this tracker.
    ///
    /// # Errors
    ///
    /// Returns an error if resource removal fails.
    pub fn remove(&self) -> Result<(), LLVMString> {
        let err = unsafe { LLVMOrcResourceTrackerRemove(self.raw) };
        into_result(err)
    }

    /// Transfers all tracked resources to another resource tracker.
    pub fn transfer_to(&self, dest: &ResourceTracker<'jit>) {
        unsafe {
            LLVMOrcResourceTrackerTransferTo(self.raw, dest.raw);
        }
    }

    /// Returns the raw `LLVMOrcResourceTrackerRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcResourceTrackerRef {
        self.raw
    }
}

impl Drop for ResourceTracker<'_> {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcReleaseResourceTracker(self.raw);
        }
    }
}

/// A symbol definition generator attached to a JITDylib.
///
/// Generators are called when a symbol lookup fails, allowing on-demand symbol
/// introduction. Consumed when added to a JITDylib via
/// [`JitDylib::add_generator()`].
pub struct DefinitionGenerator {
    raw: LLVMOrcDefinitionGeneratorRef,
}

impl DefinitionGenerator {
    /// Creates a generator that searches the host process for symbols.
    ///
    /// This allows JIT-compiled code to call `printf`, `malloc`, and any other
    /// symbol available in the host process.
    pub fn for_process(global_prefix: char) -> Result<Self, LLVMString> {
        let mut raw = std::ptr::null_mut();
        let err = unsafe {
            LLVMOrcCreateDynamicLibrarySearchGeneratorForProcess(
                &mut raw,
                global_prefix as libc::c_char,
                None,
                std::ptr::null_mut(),
            )
        };
        into_result(err)?;
        Ok(DefinitionGenerator { raw })
    }

    /// Creates a generator that searches a dynamic library at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded.
    pub fn for_dynamic_library(path: &str, global_prefix: char) -> Result<Self, LLVMString> {
        let c_path = to_c_str(path);
        let mut raw = std::ptr::null_mut();
        let err = unsafe {
            LLVMOrcCreateDynamicLibrarySearchGeneratorForPath(
                &mut raw,
                c_path.as_ptr(),
                global_prefix as libc::c_char,
                None,
                std::ptr::null_mut(),
            )
        };
        into_result(err)?;
        Ok(DefinitionGenerator { raw })
    }

    /// Creates a generator that searches a static library at the given path.
    ///
    /// # Errors
    ///
    /// Returns an error if the library cannot be loaded.
    pub fn for_static_library(
        obj_layer: &ObjectLayer<'_>,
        path: &str,
        target_triple: &str,
    ) -> Result<Self, LLVMString> {
        let c_path = to_c_str(path);
        let c_triple = to_c_str(target_triple);
        let mut raw = std::ptr::null_mut();
        let err = unsafe {
            LLVMOrcCreateStaticLibrarySearchGeneratorForPath(
                &mut raw,
                obj_layer.raw,
                c_path.as_ptr(),
                c_triple.as_ptr(),
            )
        };
        into_result(err)?;
        Ok(DefinitionGenerator { raw })
    }

    /// Creates a custom definition generator with a user-provided callback.
    ///
    /// # Safety
    ///
    /// The callback, context, and dispose function must be valid and correctly
    /// implemented.
    pub unsafe fn custom(
        try_to_generate: LLVMOrcCAPIDefinitionGeneratorTryToGenerateFunction,
        ctx: *mut libc::c_void,
        dispose: LLVMOrcDisposeCAPIDefinitionGeneratorFunction,
    ) -> Self {
        unsafe {
            let raw = LLVMOrcCreateCustomCAPIDefinitionGenerator(try_to_generate, ctx, dispose);
            DefinitionGenerator { raw }
        }
    }

    fn into_raw(self) -> LLVMOrcDefinitionGeneratorRef {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

impl Drop for DefinitionGenerator {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeDefinitionGenerator(self.raw);
        }
    }
}

impl Debug for DefinitionGenerator {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DefinitionGenerator").finish_non_exhaustive()
    }
}

/// A bundle of symbol definitions that share a common materialization process.
///
/// Created by [`absolute_symbols()`], [`lazy_reexports()`], or
/// [`custom_materialization_unit()`].
pub struct MaterializationUnit {
    raw: LLVMOrcMaterializationUnitRef,
}

impl MaterializationUnit {
    fn into_raw(self) -> LLVMOrcMaterializationUnitRef {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }

    /// Returns the raw `LLVMOrcMaterializationUnitRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcMaterializationUnitRef {
        self.raw
    }
}

impl Drop for MaterializationUnit {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeMaterializationUnit(self.raw);
        }
    }
}

impl Debug for MaterializationUnit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaterializationUnit").finish_non_exhaustive()
    }
}

/// Creates a materialization unit for absolute (pre-resolved) symbols.
///
/// Use this to define symbols with known addresses, e.g., mapping runtime
/// function pointers into the JIT.
///
/// # Safety
///
/// The `syms` pointer and `num_pairs` must be valid.
pub unsafe fn absolute_symbols(syms: LLVMOrcCSymbolMapPairs, num_pairs: usize) -> MaterializationUnit {
    unsafe {
        let raw = LLVMOrcAbsoluteSymbols(syms, num_pairs);
        MaterializationUnit { raw }
    }
}

/// Creates a materialization unit for lazy reexports.
///
/// This is the core mechanism for lazy compilation: stub symbols are defined
/// that trigger compilation of the actual implementation on first call.
///
/// # Safety
///
/// All pointer parameters must be valid.
pub unsafe fn lazy_reexports(
    lctm: &LazyCallThroughManager,
    ism: &IndirectStubsManager,
    source_jd: &JitDylib<'_>,
    callable_aliases: LLVMOrcCSymbolAliasMapPairs,
    num_pairs: usize,
) -> MaterializationUnit {
    unsafe {
        let raw = LLVMOrcLazyReexports(lctm.raw, ism.raw, source_jd.raw, callable_aliases, num_pairs);
        MaterializationUnit { raw }
    }
}

/// Creates a custom materialization unit with user-provided callbacks.
///
/// # Safety
///
/// All function pointers and the context must be valid and correctly implemented.
pub unsafe fn custom_materialization_unit(
    name: &str,
    ctx: *mut libc::c_void,
    syms: LLVMOrcCSymbolFlagsMapPairs,
    num_syms: usize,
    init_sym: LLVMOrcSymbolStringPoolEntryRef,
    materialize: LLVMOrcMaterializationUnitMaterializeFunction,
    discard: LLVMOrcMaterializationUnitDiscardFunction,
    destroy: LLVMOrcMaterializationUnitDestroyFunction,
) -> MaterializationUnit {
    unsafe {
        let c_name = to_c_str(name);
        let raw = LLVMOrcCreateCustomMaterializationUnit(
            c_name.as_ptr(),
            ctx,
            syms,
            num_syms,
            init_sym,
            materialize,
            discard,
            destroy,
        );
        MaterializationUnit { raw }
    }
}

/// Represents the responsibility for materializing a set of symbols.
///
/// Received by custom materialization unit callbacks. Provides methods to
/// notify the JIT of materialization progress (resolved, emitted) or failure.
///
/// Dropping this without calling [`notify_emitted()`](Self::notify_emitted) or
/// [`fail_materialization()`](Self::fail_materialization) will fail any pending
/// queries for the symbols this responsibility covers.
pub struct MaterializationResponsibility {
    raw: LLVMOrcMaterializationResponsibilityRef,
}

impl MaterializationResponsibility {
    /// Creates a wrapper from a raw ref. Used inside materialization callbacks.
    ///
    /// # Safety
    ///
    /// The raw ref must be valid and the caller must be the materialization
    /// callback that received it.
    pub unsafe fn from_raw(raw: LLVMOrcMaterializationResponsibilityRef) -> Self {
        MaterializationResponsibility { raw }
    }

    /// Returns the target JITDylib for this materialization.
    ///
    /// The returned reference is non-owning and borrows from `self`.
    pub fn get_target_dylib(&self) -> LLVMOrcJITDylibRef {
        unsafe { LLVMOrcMaterializationResponsibilityGetTargetDylib(self.raw) }
    }

    /// Returns the execution session for this materialization.
    pub fn get_execution_session(&self) -> LLVMOrcExecutionSessionRef {
        unsafe { LLVMOrcMaterializationResponsibilityGetExecutionSession(self.raw) }
    }

    /// Returns the symbols that this responsibility covers.
    ///
    /// The returned pairs and count must be freed with
    /// [`dispose_symbol_flags_map()`].
    pub fn get_symbols(&self) -> (LLVMOrcCSymbolFlagsMapPairs, usize) {
        let mut num_pairs: libc::size_t = 0;
        let pairs = unsafe { LLVMOrcMaterializationResponsibilityGetSymbols(self.raw, &mut num_pairs) };
        (pairs, num_pairs)
    }

    /// Returns the initializer symbol for this responsibility, if any.
    pub fn get_initializer_symbol(&self) -> LLVMOrcSymbolStringPoolEntryRef {
        unsafe { LLVMOrcMaterializationResponsibilityGetInitializerSymbol(self.raw) }
    }

    /// Returns the symbols that triggered this materialization.
    ///
    /// The returned array and count must be freed with [`dispose_symbols()`].
    pub fn get_requested_symbols(&self) -> (*mut LLVMOrcSymbolStringPoolEntryRef, usize) {
        let mut num_symbols: libc::size_t = 0;
        let syms = unsafe { LLVMOrcMaterializationResponsibilityGetRequestedSymbols(self.raw, &mut num_symbols) };
        (syms, num_symbols)
    }

    /// Notifies the JIT that the given symbols have been resolved to addresses.
    ///
    /// # Errors
    ///
    /// Returns an error if resolution fails.
    ///
    /// # Safety
    ///
    /// The `symbols` pointer and `num_pairs` must be valid.
    pub unsafe fn notify_resolved(&self, symbols: LLVMOrcCSymbolMapPairs, num_pairs: usize) -> Result<(), LLVMString> {
        unsafe {
            let err = LLVMOrcMaterializationResponsibilityNotifyResolved(self.raw, symbols, num_pairs);
            into_result(err)
        }
    }

    /// Notifies the JIT that materialization is complete and symbols are safe to use.
    ///
    /// On LLVM 18, takes no dependency group parameters.
    /// On LLVM 19+, takes symbol dependency groups.
    ///
    /// # Errors
    ///
    /// Returns an error if emission fails.
    #[llvm_versions(..19)]
    pub unsafe fn notify_emitted(&self) -> Result<(), LLVMString> {
        let err = LLVMOrcMaterializationResponsibilityNotifyEmitted(self.raw);
        into_result(err)
    }

    /// Notifies the JIT that materialization is complete and symbols are safe to use.
    ///
    /// On LLVM 19+, requires symbol dependency groups.
    ///
    /// # Errors
    ///
    /// Returns an error if emission fails.
    ///
    /// # Safety
    ///
    /// The `symbol_dep_groups` pointer and count must be valid.
    #[llvm_versions(19..)]
    pub unsafe fn notify_emitted(
        &self,
        symbol_dep_groups: *mut LLVMOrcCSymbolDependenceGroup,
        num_symbol_dep_groups: usize,
    ) -> Result<(), LLVMString> {
        unsafe {
            let err =
                LLVMOrcMaterializationResponsibilityNotifyEmitted(self.raw, symbol_dep_groups, num_symbol_dep_groups);
            into_result(err)
        }
    }

    /// Claims responsibility for additional symbols discovered during
    /// materialization.
    ///
    /// # Safety
    ///
    /// The `pairs` pointer and `num_pairs` must be valid.
    pub unsafe fn define_materializing(
        &self,
        pairs: LLVMOrcCSymbolFlagsMapPairs,
        num_pairs: usize,
    ) -> Result<(), LLVMString> {
        unsafe {
            let err = LLVMOrcMaterializationResponsibilityDefineMaterializing(self.raw, pairs, num_pairs);
            into_result(err)
        }
    }

    /// Reports that materialization has failed.
    ///
    /// All symbols covered by this responsibility will transition to an error
    /// state, and pending queries will receive a failure notification.
    ///
    /// Consumes `self`.
    pub fn fail_materialization(self) {
        unsafe {
            LLVMOrcMaterializationResponsibilityFailMaterialization(self.raw);
        }
        std::mem::forget(self); // consumed by LLVM, skip Drop
    }

    /// Hands symbols back to a different materialization unit.
    ///
    /// Consumes the given `MaterializationUnit`.
    ///
    /// # Errors
    ///
    /// Returns an error if replacement fails.
    pub fn replace(&self, mu: MaterializationUnit) -> Result<(), LLVMString> {
        let err = unsafe { LLVMOrcMaterializationResponsibilityReplace(self.raw, mu.into_raw()) };
        into_result(err)
    }

    /// Delegates responsibility for a subset of symbols to a new
    /// `MaterializationResponsibility`.
    ///
    /// # Safety
    ///
    /// The `symbols` pointer and `num_symbols` must be valid.
    pub unsafe fn delegate(
        &self,
        symbols: *mut LLVMOrcSymbolStringPoolEntryRef,
        num_symbols: usize,
    ) -> Result<MaterializationResponsibility, LLVMString> {
        unsafe {
            let mut result = std::ptr::null_mut();
            let err = LLVMOrcMaterializationResponsibilityDelegate(self.raw, symbols, num_symbols, &mut result);
            into_result(err)?;
            Ok(MaterializationResponsibility { raw: result })
        }
    }

    /// Returns the raw `LLVMOrcMaterializationResponsibilityRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcMaterializationResponsibilityRef {
        self.raw
    }
}

impl Drop for MaterializationResponsibility {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeMaterializationResponsibility(self.raw);
        }
    }
}

impl Debug for MaterializationResponsibility {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("MaterializationResponsibility").finish_non_exhaustive()
    }
}

/// Frees a symbol flags map returned by
/// [`MaterializationResponsibility::get_symbols()`].
///
/// # Safety
///
/// The pointer must have been returned by `get_symbols()`.
pub unsafe fn dispose_symbol_flags_map(pairs: LLVMOrcCSymbolFlagsMapPairs) {
    unsafe {
        LLVMOrcDisposeCSymbolFlagsMap(pairs);
    }
}

/// Frees a symbols array returned by
/// [`MaterializationResponsibility::get_requested_symbols()`].
///
/// # Safety
///
/// The pointer must have been returned by `get_requested_symbols()`.
pub unsafe fn dispose_symbols(symbols: *mut LLVMOrcSymbolStringPoolEntryRef) {
    unsafe {
        LLVMOrcDisposeSymbols(symbols);
    }
}

/// Captures target configuration for JIT compilation.
///
/// Can auto-detect the host or be created from an existing `TargetMachine`.
#[derive(PartialEq, Eq)]
pub struct JitTargetMachineBuilder {
    raw: LLVMOrcJITTargetMachineBuilderRef,
}

impl JitTargetMachineBuilder {
    /// Detects the host and creates a target machine builder for it.
    ///
    /// # Errors
    ///
    /// Returns an error if host detection fails.
    pub fn detect_host() -> Result<Self, LLVMString> {
        let mut raw = std::ptr::null_mut();
        let err = unsafe { LLVMOrcJITTargetMachineBuilderDetectHost(&mut raw) };
        into_result(err)?;
        Ok(JitTargetMachineBuilder { raw })
    }

    /// Creates a builder from an existing target machine.
    ///
    /// # Safety
    ///
    /// The `TargetMachine` is consumed. The caller must not use it after.
    pub unsafe fn from_target_machine(tm: llvm_sys::target_machine::LLVMTargetMachineRef) -> Self {
        unsafe {
            let raw = LLVMOrcJITTargetMachineBuilderCreateFromTargetMachine(tm);
            JitTargetMachineBuilder { raw }
        }
    }

    /// Returns the target triple string.
    pub fn get_target_triple(&self) -> String {
        unsafe {
            let ptr = LLVMOrcJITTargetMachineBuilderGetTargetTriple(self.raw);
            let s = CStr::from_ptr(ptr).to_string_lossy().into_owned();
            llvm_sys::core::LLVMDisposeMessage(ptr);
            s
        }
    }

    /// Sets the target triple.
    pub fn set_target_triple(&mut self, triple: &str) {
        let c_triple = to_c_str(triple);
        unsafe {
            LLVMOrcJITTargetMachineBuilderSetTargetTriple(self.raw, c_triple.as_ptr());
        }
    }

    pub(crate) fn into_raw(self) -> LLVMOrcJITTargetMachineBuilderRef {
        let raw = self.raw;
        std::mem::forget(self);
        raw
    }
}

impl Drop for JitTargetMachineBuilder {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeJITTargetMachineBuilder(self.raw);
        }
    }
}

impl Debug for JitTargetMachineBuilder {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("JitTargetMachineBuilder")
            .field("triple", &self.get_target_triple())
            .finish()
    }
}

/// A non-owning reference to an object linking layer.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectLayer<'jit> {
    raw: LLVMOrcObjectLayerRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl<'jit> ObjectLayer<'jit> {
    /// Adds an object file to a JITDylib.
    ///
    /// Consumes the memory buffer.
    pub fn add_object_file(&self, jd: &JitDylib<'jit>, obj_buffer: MemoryBuffer) -> Result<(), LLVMString> {
        let raw_buf = obj_buffer.memory_buffer;
        std::mem::forget(obj_buffer);
        let err = unsafe { LLVMOrcObjectLayerAddObjectFile(self.raw, jd.raw, raw_buf) };
        into_result(err)
    }

    /// Adds an object file to a JITDylib with a specific resource tracker.
    ///
    /// Consumes the memory buffer.
    pub fn add_object_file_with_rt(
        &self,
        rt: &ResourceTracker<'jit>,
        obj_buffer: MemoryBuffer,
    ) -> Result<(), LLVMString> {
        let raw_buf = obj_buffer.memory_buffer;
        std::mem::forget(obj_buffer);
        let err = unsafe { LLVMOrcObjectLayerAddObjectFileWithRT(self.raw, rt.raw, raw_buf) };
        into_result(err)
    }

    /// Emits an object buffer through this layer for the given materialization
    /// responsibility.
    ///
    /// # Safety
    ///
    /// The materialization responsibility must be valid and match the object.
    pub unsafe fn emit(&self, mr: MaterializationResponsibility, obj_buffer: MemoryBuffer) {
        unsafe {
            let raw_buf = obj_buffer.memory_buffer;
            std::mem::forget(obj_buffer);
            let mr_raw = mr.raw;
            std::mem::forget(mr); // consumed by LLVM
            LLVMOrcObjectLayerEmit(self.raw, mr_raw, raw_buf);
        }
    }

    /// Returns the raw `LLVMOrcObjectLayerRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcObjectLayerRef {
        self.raw
    }
}

/// A non-owning reference to an IR transform layer.
///
/// Allows setting a callback that transforms IR modules before compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct IRTransformLayer<'jit> {
    raw: LLVMOrcIRTransformLayerRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl<'jit> IRTransformLayer<'jit> {
    /// Sets the IR transform callback.
    ///
    /// The callback is invoked on each module before it is compiled. Use this
    /// to run optimization passes.
    ///
    /// # Safety
    ///
    /// The callback and context must be valid for the lifetime of the LLJIT.
    pub unsafe fn set_transform(&self, transform: LLVMOrcIRTransformLayerTransformFunction, ctx: *mut libc::c_void) {
        unsafe {
            LLVMOrcIRTransformLayerSetTransform(self.raw, transform, ctx);
        }
    }

    /// Emits a thread-safe module through this layer for the given
    /// materialization responsibility.
    ///
    /// Consumes both the materialization responsibility and the module.
    ///
    /// # Safety
    ///
    /// The materialization responsibility must be valid and correspond to the
    /// symbols in the module.
    pub unsafe fn emit(&self, mr: MaterializationResponsibility, tsm: ThreadSafeModule) {
        unsafe {
            let mr_raw = mr.raw;
            std::mem::forget(mr); // consumed by LLVM
            LLVMOrcIRTransformLayerEmit(self.raw, mr_raw, tsm.into_raw());
        }
    }

    /// Returns the raw `LLVMOrcIRTransformLayerRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcIRTransformLayerRef {
        self.raw
    }
}

/// A non-owning reference to an object transform layer.
///
/// Allows setting a callback that transforms object files after compilation.
#[derive(Debug, Clone, Copy, PartialEq, Eq)]
pub struct ObjectTransformLayer<'jit> {
    raw: LLVMOrcObjectTransformLayerRef,
    _marker: PhantomData<&'jit LLJit>,
}

impl<'jit> ObjectTransformLayer<'jit> {
    /// Sets the object transform callback.
    ///
    /// The callback is invoked on each compiled object file before linking.
    /// Use this for dumping objects to disk or post-compilation instrumentation.
    ///
    /// # Safety
    ///
    /// The callback and context must be valid for the lifetime of the LLJIT.
    pub unsafe fn set_transform(
        &self,
        transform: LLVMOrcObjectTransformLayerTransformFunction,
        ctx: *mut libc::c_void,
    ) {
        unsafe {
            LLVMOrcObjectTransformLayerSetTransform(self.raw, transform, ctx);
        }
    }

    /// Returns the raw `LLVMOrcObjectTransformLayerRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcObjectTransformLayerRef {
        self.raw
    }
}

/// Manages indirect stubs for lazy compilation.
///
/// Stubs are small code fragments that read a function pointer and jump to it.
/// Initially they point to lazy call-through entries that trigger compilation.
pub struct IndirectStubsManager {
    raw: LLVMOrcIndirectStubsManagerRef,
}

impl IndirectStubsManager {
    /// Creates an indirect stubs manager for the given target triple.
    pub fn create(target_triple: &str) -> Self {
        let c_triple = to_c_str(target_triple);
        let raw = unsafe { LLVMOrcCreateLocalIndirectStubsManager(c_triple.as_ptr()) };
        IndirectStubsManager { raw }
    }

    /// Returns the raw `LLVMOrcIndirectStubsManagerRef`.
    pub unsafe fn as_raw(&self) -> LLVMOrcIndirectStubsManagerRef {
        self.raw
    }
}

impl Drop for IndirectStubsManager {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeIndirectStubsManager(self.raw);
        }
    }
}

impl Debug for IndirectStubsManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("IndirectStubsManager").finish_non_exhaustive()
    }
}

/// Manages trampolines that trigger lazy compilation on first call.
///
/// Works with [`IndirectStubsManager`] and [`lazy_reexports()`] to implement
/// lazy compilation.
pub struct LazyCallThroughManager {
    raw: LLVMOrcLazyCallThroughManagerRef,
}

impl LazyCallThroughManager {
    /// Creates a lazy call-through manager.
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails.
    pub fn create(
        target_triple: &str,
        es: &ExecutionSession<'_>,
        error_handler_addr: LLVMOrcJITTargetAddress,
    ) -> Result<Self, LLVMString> {
        let c_triple = to_c_str(target_triple);
        let mut raw = std::ptr::null_mut();
        let err = unsafe {
            LLVMOrcCreateLocalLazyCallThroughManager(c_triple.as_ptr(), es.raw, error_handler_addr, &mut raw)
        };
        into_result(err)?;
        Ok(LazyCallThroughManager { raw })
    }

    /// Returns the raw `LLVMOrcLazyCallThroughManagerRef`.
    pub unsafe fn as_raw(&self) -> LLVMOrcLazyCallThroughManagerRef {
        self.raw
    }
}

impl Drop for LazyCallThroughManager {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeLazyCallThroughManager(self.raw);
        }
    }
}

impl Debug for LazyCallThroughManager {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LazyCallThroughManager").finish_non_exhaustive()
    }
}

/// Helper for dumping JIT'd object files to disk.
///
/// Use with [`ObjectTransformLayer::set_transform()`] to save compiled objects
/// for debugging.
pub struct DumpObjects {
    raw: LLVMOrcDumpObjectsRef,
}

impl DumpObjects {
    /// Creates a dump objects helper.
    ///
    /// `dump_dir`: directory to write object files to.
    /// `identifier_override`: optional filename override (empty string for default).
    pub fn create(dump_dir: &str, identifier_override: &str) -> Self {
        let c_dir = to_c_str(dump_dir);
        let c_id = to_c_str(identifier_override);
        let raw = unsafe { LLVMOrcCreateDumpObjects(c_dir.as_ptr(), c_id.as_ptr()) };
        DumpObjects { raw }
    }

    /// Invokes the dump on an object buffer.
    ///
    /// # Safety
    ///
    /// The buffer pointer must be valid.
    pub unsafe fn call(&self, obj_buffer: *mut LLVMMemoryBufferRef) -> Result<(), LLVMString> {
        unsafe {
            let err = LLVMOrcDumpObjects_CallOperator(self.raw, obj_buffer);
            into_result(err)
        }
    }

    /// Returns the raw `LLVMOrcDumpObjectsRef`.
    pub unsafe fn as_raw(&self) -> LLVMOrcDumpObjectsRef {
        self.raw
    }
}

impl Drop for DumpObjects {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeDumpObjects(self.raw);
        }
    }
}

impl Debug for DumpObjects {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("DumpObjects").finish_non_exhaustive()
    }
}

/// Represents a suspended symbol lookup that can be resumed asynchronously.
///
/// Used by custom definition generators that need to perform async work before
/// providing symbol definitions.
pub struct LookupState {
    raw: LLVMOrcLookupStateRef,
}

impl LookupState {
    /// Resumes a suspended lookup.
    ///
    /// Pass a null error to indicate success, or an error to fail the lookup.
    pub fn continue_lookup(self, err: llvm_sys::error::LLVMErrorRef) {
        unsafe {
            LLVMOrcLookupStateContinueLookup(self.raw, err);
        }
        std::mem::forget(self); // consumed by LLVM
    }
}

impl Debug for LookupState {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LookupState").finish_non_exhaustive()
    }
}

/// Builder for configuring and creating an [`LLJit`] instance.
///
/// If no customization is needed, use [`LLJit::create()`] which uses defaults.
#[derive(Debug, PartialEq, Eq)]
pub struct LLJitBuilder {
    raw: LLVMOrcLLJITBuilderRef,
}

impl LLJitBuilder {
    /// Creates a new LLJIT builder with default settings.
    pub fn create() -> Self {
        let raw = unsafe { LLVMOrcCreateLLJITBuilder() };
        assert!(!raw.is_null());
        LLJitBuilder { raw }
    }

    /// Sets the JIT target machine builder, overriding automatic host detection.
    pub fn set_jit_target_machine_builder(&mut self, jtmb: JitTargetMachineBuilder) -> &mut Self {
        unsafe {
            LLVMOrcLLJITBuilderSetJITTargetMachineBuilder(self.raw, jtmb.into_raw());
        }
        self
    }

    /// Sets a custom object linking layer creator.
    ///
    /// # Safety
    ///
    /// The callback and context must be valid for the lifetime of the LLJIT.
    pub unsafe fn set_object_linking_layer_creator(
        &mut self,
        creator: LLVMOrcLLJITBuilderObjectLinkingLayerCreatorFunction,
        ctx: *mut libc::c_void,
    ) -> &mut Self {
        unsafe {
            LLVMOrcLLJITBuilderSetObjectLinkingLayerCreator(self.raw, creator, ctx);
            self
        }
    }

    /// Builds the [`LLJit`] instance, consuming this builder.
    ///
    /// # Errors
    ///
    /// Returns an error if LLJIT creation fails.
    pub fn build(self) -> Result<LLJit, LLVMString> {
        let raw = self.raw;
        // Builder is always consumed by LLVMOrcCreateLLJIT (even on error).
        std::mem::forget(self);
        let mut lljit = std::ptr::null_mut();
        let err = unsafe { LLVMOrcCreateLLJIT(&mut lljit, raw) };
        into_result(err)?;
        assert!(!lljit.is_null());
        Ok(LLJit { raw: lljit })
    }
}

impl Drop for LLJitBuilder {
    fn drop(&mut self) {
        unsafe {
            LLVMOrcDisposeLLJITBuilder(self.raw);
        }
    }
}

/// The LLJIT JIT compiler -- a pre-assembled ORC2 compilation pipeline.
///
/// See the [module-level documentation](crate::orc2) for a full example.
pub struct LLJit {
    raw: LLVMOrcLLJITRef,
}

impl LLJit {
    /// Creates an LLJIT instance with default settings.
    ///
    /// Call [`Target::initialize_native()`](crate::targets::Target::initialize_native)
    /// before this.
    ///
    /// # Errors
    ///
    /// Returns an error if creation fails.
    pub fn create() -> Result<Self, LLVMString> {
        LLJitBuilder::create().build()
    }

    /// Returns the execution session.
    pub fn execution_session(&self) -> ExecutionSession<'_> {
        let raw = unsafe { LLVMOrcLLJITGetExecutionSession(self.raw) };
        assert!(!raw.is_null());
        ExecutionSession {
            raw,
            _marker: PhantomData,
        }
    }

    /// Returns the main JITDylib.
    pub fn main_jit_dylib(&self) -> JitDylib<'_> {
        let raw = unsafe { LLVMOrcLLJITGetMainJITDylib(self.raw) };
        assert!(!raw.is_null());
        JitDylib {
            raw,
            _marker: PhantomData,
        }
    }

    /// Returns the target triple string.
    pub fn get_triple_string(&self) -> &str {
        unsafe {
            let ptr = LLVMOrcLLJITGetTripleString(self.raw);
            CStr::from_ptr(ptr).to_str().expect("LLJIT triple is not valid UTF-8")
        }
    }

    /// Returns the global symbol prefix character for the target platform.
    pub fn get_global_prefix(&self) -> char {
        let ch = unsafe { LLVMOrcLLJITGetGlobalPrefix(self.raw) };
        ch as u8 as char
    }

    /// Returns the data layout string.
    pub fn get_data_layout_str(&self) -> &str {
        unsafe {
            let ptr = LLVMOrcLLJITGetDataLayoutStr(self.raw);
            CStr::from_ptr(ptr)
                .to_str()
                .expect("LLJIT data layout is not valid UTF-8")
        }
    }

    /// Mangles and interns a symbol name for the target.
    ///
    /// Returns an interned symbol that accounts for platform-specific mangling
    /// (e.g., `_` prefix on macOS).
    pub fn mangle_and_intern(&self, name: &str) -> SymbolStringPoolEntry {
        let c_name = to_c_str(name);
        let raw = unsafe { LLVMOrcLLJITMangleAndIntern(self.raw, c_name.as_ptr()) };
        SymbolStringPoolEntry { raw }
    }

    /// Adds a [`ThreadSafeModule`] to the given JITDylib.
    ///
    /// Consumes the module. Compilation is deferred until a symbol is looked up.
    ///
    /// # Errors
    ///
    /// Returns an error if the module cannot be added.
    pub fn add_module(&self, jit_dylib: &JitDylib<'_>, module: ThreadSafeModule) -> Result<(), LLVMString> {
        let err = unsafe { LLVMOrcLLJITAddLLVMIRModule(self.raw, jit_dylib.raw, module.into_raw()) };
        into_result(err)
    }

    /// Adds a [`ThreadSafeModule`] with a specific [`ResourceTracker`].
    ///
    /// Like [`add_module()`](Self::add_module), but enables later removal via
    /// [`ResourceTracker::remove()`].
    pub fn add_module_with_rt(&self, rt: &ResourceTracker<'_>, module: ThreadSafeModule) -> Result<(), LLVMString> {
        let err = unsafe { LLVMOrcLLJITAddLLVMIRModuleWithRT(self.raw, rt.raw, module.into_raw()) };
        into_result(err)
    }

    /// Adds a pre-compiled object file to the given JITDylib.
    ///
    /// Consumes the memory buffer.
    pub fn add_object_file(&self, jit_dylib: &JitDylib<'_>, obj_buffer: MemoryBuffer) -> Result<(), LLVMString> {
        let raw_buf = obj_buffer.memory_buffer;
        std::mem::forget(obj_buffer);
        let err = unsafe { LLVMOrcLLJITAddObjectFile(self.raw, jit_dylib.raw, raw_buf) };
        into_result(err)
    }

    /// Adds a pre-compiled object file with a specific [`ResourceTracker`].
    pub fn add_object_file_with_rt(
        &self,
        rt: &ResourceTracker<'_>,
        obj_buffer: MemoryBuffer,
    ) -> Result<(), LLVMString> {
        let raw_buf = obj_buffer.memory_buffer;
        std::mem::forget(obj_buffer);
        let err = unsafe { LLVMOrcLLJITAddObjectFileWithRT(self.raw, rt.raw, raw_buf) };
        into_result(err)
    }

    /// Looks up a symbol by name and returns its address.
    ///
    /// Triggers compilation of any uncompiled code needed to resolve the symbol.
    ///
    /// # Errors
    ///
    /// Returns an error if the symbol is not found or compilation fails.
    pub fn lookup(&self, name: &str) -> Result<LLVMOrcExecutorAddress, LLVMString> {
        let c_name = to_c_str(name);
        let mut addr: LLVMOrcExecutorAddress = 0;
        let err = unsafe { LLVMOrcLLJITLookup(self.raw, &mut addr, c_name.as_ptr()) };
        into_result(err)?;
        Ok(addr)
    }

    /// Looks up a symbol and returns a type-safe function handle.
    ///
    /// # Safety
    ///
    /// The type `F` must match the actual signature of the JIT-compiled function.
    pub unsafe fn get_function<F>(&self, name: &str) -> Result<LLJitFunction<'_, F>, LLVMString>
    where
        F: UnsafeFunctionPointer,
    {
        unsafe {
            let addr = self.lookup(name)?;
            assert_eq!(
                size_of::<F>(),
                size_of::<usize>(),
                "The type `F` must have the same size as a function pointer"
            );
            Ok(LLJitFunction {
                inner: transmute_copy(&(addr as usize)),
                _marker: PhantomData,
            })
        }
    }

    /// Returns the object linking layer.
    pub fn get_obj_linking_layer(&self) -> ObjectLayer<'_> {
        let raw = unsafe { LLVMOrcLLJITGetObjLinkingLayer(self.raw) };
        assert!(!raw.is_null());
        ObjectLayer {
            raw,
            _marker: PhantomData,
        }
    }

    /// Returns the object transform layer.
    pub fn get_obj_transform_layer(&self) -> ObjectTransformLayer<'_> {
        let raw = unsafe { LLVMOrcLLJITGetObjTransformLayer(self.raw) };
        assert!(!raw.is_null());
        ObjectTransformLayer {
            raw,
            _marker: PhantomData,
        }
    }

    /// Returns the IR transform layer.
    pub fn get_ir_transform_layer(&self) -> IRTransformLayer<'_> {
        let raw = unsafe { LLVMOrcLLJITGetIRTransformLayer(self.raw) };
        assert!(!raw.is_null());
        IRTransformLayer {
            raw,
            _marker: PhantomData,
        }
    }

    /// Adds a host process symbol generator to the given JITDylib.
    ///
    /// Convenience method combining [`DefinitionGenerator::for_process()`]
    /// and [`JitDylib::add_generator()`].
    pub fn add_process_symbols_generator(&self, jit_dylib: &JitDylib<'_>) -> Result<(), LLVMString> {
        let prefix = self.get_global_prefix();
        let def_gen = DefinitionGenerator::for_process(prefix)?;
        jit_dylib.add_generator(def_gen);
        Ok(())
    }

    /// Returns the raw `LLVMOrcLLJITRef`.
    ///
    /// # Safety
    ///
    /// The caller must not dispose of the returned reference.
    pub unsafe fn as_raw(&self) -> LLVMOrcLLJITRef {
        self.raw
    }
}

impl Debug for LLJit {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_struct("LLJit")
            .field("triple", &self.get_triple_string())
            .finish()
    }
}

impl Drop for LLJit {
    fn drop(&mut self) {
        unsafe {
            let err = LLVMOrcDisposeLLJIT(self.raw);
            if !err.is_null() {
                let msg = LLVMGetErrorMessage(err);
                eprintln!(
                    "Warning: LLVMOrcDisposeLLJIT failed: {}",
                    CStr::from_ptr(msg).to_string_lossy()
                );
                llvm_sys::error::LLVMDisposeErrorMessage(msg);
            }
        }
    }
}

/// A type-safe wrapper around a JIT-compiled function pointer.
///
/// Borrows from [`LLJit`] to prevent use-after-free.
/// Created by [`LLJit::get_function()`].
#[derive(Clone)]
pub struct LLJitFunction<'jit, F> {
    inner: F,
    _marker: PhantomData<&'jit LLJit>,
}

impl<F: Copy> LLJitFunction<'_, F> {
    /// Returns the raw function pointer.
    ///
    /// # Safety
    ///
    /// The [`LLJit`] must still be alive and the function's resources must not
    /// have been removed.
    pub unsafe fn into_raw(self) -> F {
        self.inner
    }

    /// Returns a copy of the raw function pointer.
    ///
    /// # Safety
    ///
    /// Same requirements as [`into_raw()`](Self::into_raw).
    pub unsafe fn as_raw(&self) -> F {
        self.inner
    }
}

impl<F> Debug for LLJitFunction<'_, F> {
    fn fmt(&self, f: &mut Formatter<'_>) -> fmt::Result {
        f.debug_tuple("LLJitFunction").field(&"<fn>").finish()
    }
}

macro_rules! impl_lljit_fn {
    (@recurse $first:ident $( , $rest:ident )*) => {
        impl_lljit_fn!($( $rest ),*);
    };
    (@recurse) => {};
    ($( $param:ident ),*) => {
        impl<Output, $( $param ),*> LLJitFunction<'_, unsafe extern "C" fn($( $param ),*) -> Output> {
            /// Calls the JIT-compiled function.
            ///
            /// # Safety
            ///
            /// The argument and return types must match the actual function signature.
            /// The [`LLJit`] must still be alive. The function must not have been removed.
            #[allow(non_snake_case)]
            #[inline(always)]
            pub unsafe fn call(&self, $( $param: $param ),*) -> Output { unsafe {
                (self.inner)($( $param ),*)
            }}
        }
        impl_lljit_fn!(@recurse $( $param ),*);
    };
}

impl_lljit_fn!(A, B, C, D, E, F, G, H, I, J, K, L, M);

/// Creates an RTDyld object linking layer with a section memory manager.
///
/// # Safety
///
/// The execution session must be valid.
pub unsafe fn create_rtdyld_object_linking_layer_with_section_memory_manager(
    es: &ExecutionSession<'_>,
) -> LLVMOrcObjectLayerRef {
    unsafe { LLVMOrcCreateRTDyldObjectLinkingLayerWithSectionMemoryManager(es.raw) }
}

/// Creates an RTDyld object linking layer with MCJIT-compatible memory manager callbacks.
///
/// # Safety
///
/// All callbacks must be valid and correctly implemented.
pub unsafe fn create_rtdyld_object_linking_layer_with_callbacks(
    es: &ExecutionSession<'_>,
    create_context: LLVMMemoryManagerCreateContextCallback,
    notify_terminating: LLVMMemoryManagerNotifyTerminatingCallback,
    allocate_code_section: llvm_sys::execution_engine::LLVMMemoryManagerAllocateCodeSectionCallback,
    allocate_data_section: llvm_sys::execution_engine::LLVMMemoryManagerAllocateDataSectionCallback,
    finalize_memory: llvm_sys::execution_engine::LLVMMemoryManagerFinalizeMemoryCallback,
    destroy: llvm_sys::execution_engine::LLVMMemoryManagerDestroyCallback,
) -> LLVMOrcObjectLayerRef {
    unsafe {
        LLVMOrcCreateRTDyldObjectLinkingLayerWithMCJITMemoryManagerLikeCallbacks(
            es.raw,
            create_context,
            notify_terminating,
            allocate_code_section,
            allocate_data_section,
            finalize_memory,
            destroy,
        )
    }
}

/// Registers a JIT event listener with an RTDyld object linking layer.
///
/// # Safety
///
/// The layer must be an RTDyld layer. The listener must be valid.
pub unsafe fn rtdyld_object_linking_layer_register_jit_event_listener(
    layer: LLVMOrcObjectLayerRef,
    listener: llvm_sys::prelude::LLVMJITEventListenerRef,
) {
    unsafe {
        LLVMOrcRTDyldObjectLinkingLayerRegisterJITEventListener(layer, listener);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::targets::{InitializationConfig, Target};

    type SumFunc = unsafe extern "C" fn(u64, u64, u64) -> u64;

    #[test]
    fn test_lljit_sum() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let tsc = ThreadSafeContext::create();
        let ctx = tsc.context();
        let module = ctx.create_module("test_sum");
        let builder = ctx.create_builder();

        let i64_type = ctx.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into(), i64_type.into()], false);
        let function = module.add_function("sum", fn_type, None);
        let bb = ctx.append_basic_block(function, "entry");

        builder.position_at_end(bb);
        let x = function.get_nth_param(0).unwrap().into_int_value();
        let y = function.get_nth_param(1).unwrap().into_int_value();
        let z = function.get_nth_param(2).unwrap().into_int_value();
        let sum = builder.build_int_add(x, y, "sum").unwrap();
        let sum = builder.build_int_add(sum, z, "sum").unwrap();
        builder.build_return(Some(&sum)).unwrap();

        let tsm = tsc.create_thread_safe_module(module).unwrap();

        let lljit = LLJit::create().expect("Failed to create LLJIT");
        lljit
            .add_module(&lljit.main_jit_dylib(), tsm)
            .expect("Failed to add module");

        let sum_fn = unsafe { lljit.get_function::<SumFunc>("sum").unwrap() };
        unsafe {
            assert_eq!(sum_fn.call(1, 2, 3), 6);
            assert_eq!(sum_fn.call(0, 0, 0), 0);
            assert_eq!(sum_fn.call(100, 200, 300), 600);
        }
    }

    #[test]
    fn test_lljit_lookup_missing_symbol() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let tsc = ThreadSafeContext::create();
        let ctx = tsc.context();
        let module = ctx.create_module("empty");
        let void_type = ctx.void_type();
        let fn_type = void_type.fn_type(&[], false);
        let function = module.add_function("dummy", fn_type, None);
        let bb = ctx.append_basic_block(function, "entry");
        let builder = ctx.create_builder();
        builder.position_at_end(bb);
        builder.build_return(None).unwrap();

        let tsm = tsc.create_thread_safe_module(module).unwrap();

        let lljit = LLJit::create().unwrap();
        lljit.add_module(&lljit.main_jit_dylib(), tsm).unwrap();
        assert!(lljit.lookup("nonexistent_symbol").is_err());
    }

    #[test]
    fn test_lljit_resource_tracker() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let tsc = ThreadSafeContext::create();
        let ctx = tsc.context();
        let module = ctx.create_module("rt_test");
        let builder = ctx.create_builder();
        let i64_type = ctx.i64_type();
        let fn_type = i64_type.fn_type(&[i64_type.into()], false);
        let function = module.add_function("identity", fn_type, None);
        let bb = ctx.append_basic_block(function, "entry");
        builder.position_at_end(bb);
        let x = function.get_nth_param(0).unwrap().into_int_value();
        builder.build_return(Some(&x)).unwrap();

        let tsm = tsc.create_thread_safe_module(module).unwrap();

        let lljit = LLJit::create().unwrap();
        let jd = lljit.main_jit_dylib();
        let rt = jd.create_resource_tracker();

        lljit.add_module_with_rt(&rt, tsm).unwrap();

        type IdentityFunc = unsafe extern "C" fn(u64) -> u64;
        let func = unsafe { lljit.get_function::<IdentityFunc>("identity").unwrap() };
        unsafe {
            assert_eq!(func.call(42), 42);
        }
        drop(func);

        rt.remove().unwrap();
        assert!(lljit.lookup("identity").is_err());
    }

    #[test]
    fn test_lljit_multiple_modules() {
        Target::initialize_native(&InitializationConfig::default()).unwrap();

        let tsc = ThreadSafeContext::create();
        let ctx = tsc.context();
        let builder = ctx.create_builder();
        let i64_type = ctx.i64_type();

        // Module 1: add(a, b) -> a + b
        let m1 = ctx.create_module("mod_add");
        let fn_type = i64_type.fn_type(&[i64_type.into(), i64_type.into()], false);
        let function = m1.add_function("add", fn_type, None);
        let bb = ctx.append_basic_block(function, "entry");
        builder.position_at_end(bb);
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b_val = function.get_nth_param(1).unwrap().into_int_value();
        let sum = builder.build_int_add(a, b_val, "sum").unwrap();
        builder.build_return(Some(&sum)).unwrap();

        // Module 2: mul(a, b) -> a * b
        let m2 = ctx.create_module("mod_mul");
        let function = m2.add_function("mul", fn_type, None);
        let bb = ctx.append_basic_block(function, "entry");
        builder.position_at_end(bb);
        let a = function.get_nth_param(0).unwrap().into_int_value();
        let b_val = function.get_nth_param(1).unwrap().into_int_value();
        let prod = builder.build_int_mul(a, b_val, "prod").unwrap();
        builder.build_return(Some(&prod)).unwrap();

        let tsms = vec![
            tsc.create_thread_safe_module(m1).unwrap(),
            tsc.create_thread_safe_module(m2).unwrap(),
        ];

        let lljit = LLJit::create().unwrap();
        let jd = lljit.main_jit_dylib();
        for tsm in tsms {
            lljit.add_module(&jd, tsm).unwrap();
        }

        type BinFunc = unsafe extern "C" fn(u64, u64) -> u64;
        let add_fn = unsafe { lljit.get_function::<BinFunc>("add").unwrap() };
        let mul_fn = unsafe { lljit.get_function::<BinFunc>("mul").unwrap() };
        unsafe {
            assert_eq!(add_fn.call(3, 4), 7);
            assert_eq!(mul_fn.call(3, 4), 12);
        }
    }

    #[test]
    fn test_create_thread_safe_module_rejects_wrong_context() {
        let tsc1 = ThreadSafeContext::create();
        let tsc2 = ThreadSafeContext::create();

        let ctx1 = tsc1.context();
        let module = ctx1.create_module("from_tsc1");

        // Wrapping tsc1's module with tsc2 must fail
        let result = tsc2.create_thread_safe_module(module);
        assert!(result.is_err(), "should reject module from a different context");

        // The module is returned back in the Err
        let module = result.unwrap_err();

        // Wrapping with the correct tsc1 must succeed
        let result = tsc1.create_thread_safe_module(module);
        assert!(result.is_ok(), "should accept module from the correct context");
    }
}
