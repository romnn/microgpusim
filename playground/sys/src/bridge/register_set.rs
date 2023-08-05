#[cxx::bridge]
mod ffi {
    unsafe extern "C++" {
        include!("playground-sys/src/ref/bridge/register_set.hpp");

        type register_set = crate::bridge::types::register_set::register_set;
        type warp_inst_ptr = crate::bridge::warp_inst::warp_inst_ptr;

        type register_set_ptr;
        #[must_use] fn get(self: &register_set_ptr) -> *const register_set;

        type register_set_bridge;

        #[must_use]
        unsafe fn new_register_set_bridge(
            ptr: *const register_set,
            owned: bool,
        ) -> SharedPtr<register_set_bridge>;

        #[must_use]
        fn inner(self: &register_set_bridge) -> *const register_set;

        #[must_use]
        fn get_registers(self: &register_set_bridge) -> UniquePtr<CxxVector<warp_inst_ptr>>;
    }

    // explicit instantiation for register_set_ptr to implement VecElement
    impl CxxVector<register_set_ptr> {}

    // explicit instantiation for register_set_bridge to implement VecElement
    impl CxxVector<register_set_bridge> {}
}

pub use ffi::*;
