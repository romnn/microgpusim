pub use std::sync::atomic;
pub use std::sync::Arc;

#[cfg(feature = "parking_lot")]
pub mod parking_lot {
    /// A mutex
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(parking_lot::Mutex<T>);

    impl<T> Mutex<T> {
        #[must_use]
        #[inline]
        pub fn new(value: T) -> Self {
            Self(parking_lot::Mutex::new(value))
        }
    }

    impl<T: ?Sized> Mutex<T> {
        #[must_use]
        #[inline]
        pub fn lock(&self) -> parking_lot::MutexGuard<T> {
            self.0.lock()
        }

        #[must_use]
        #[inline]
        pub fn try_lock(&self) -> parking_lot::MutexGuard<T> {
            self.0.try_lock().unwrap()
        }
    }

    /// A read-write lock
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(parking_lot::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        #[inline]
        pub fn new(value: T) -> RwLock<T> {
            Self(parking_lot::RwLock::new(value))
        }

        #[must_use]
        #[inline]
        pub fn read(&self) -> parking_lot::RwLockReadGuard<T> {
            self.0.read()
        }

        #[must_use]
        #[inline]
        pub fn try_read(&self) -> parking_lot::RwLockReadGuard<T> {
            self.0.try_read().unwrap()
        }

        #[must_use]
        #[inline]
        pub fn write(&self) -> parking_lot::RwLockWriteGuard<T> {
            self.0.write()
        }

        #[must_use]
        #[inline]
        pub fn try_write(&self) -> parking_lot::RwLockWriteGuard<T> {
            self.0.try_write().unwrap()
        }
    }
}

#[cfg(not(feature = "parking_lot"))]
pub mod std {
    /// A mutex
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct Mutex<T: ?Sized>(std::sync::Mutex<T>);

    impl<T> Mutex<T> {
        #[must_use]
        #[inline]
        pub fn new(value: T) -> Self {
            Self(std::sync::Mutex::new(value))
        }
    }

    impl<T: ?Sized> Mutex<T> {
        #[must_use]
        #[inline]
        pub fn lock(&self) -> std::sync::MutexGuard<T> {
            self.0.lock().unwrap()
        }

        #[must_use]
        #[inline]
        pub fn try_lock(&self) -> std::sync::MutexGuard<T> {
            self.0.try_lock().unwrap()
        }
    }

    /// A read-write lock
    #[repr(transparent)]
    #[derive(Debug, Default)]
    pub struct RwLock<T: ?Sized>(std::sync::RwLock<T>);

    impl<T> RwLock<T> {
        #[must_use]
        #[inline]
        pub fn new(value: T) -> RwLock<T> {
            Self(std::sync::RwLock::new(value))
        }

        #[must_use]
        #[inline]
        pub fn read(&self) -> std::sync::RwLockReadGuard<T> {
            self.0.read().unwrap()
        }

        #[must_use]
        #[inline]
        pub fn try_read(&self) -> std::sync::RwLockReadGuard<T> {
            self.0.try_read().unwrap()
        }

        #[must_use]
        #[inline]
        pub fn write(&self) -> std::sync::RwLockWriteGuard<T> {
            self.0.write().unwrap()
        }

        #[must_use]
        #[inline]
        pub fn try_write(&self) -> std::sync::RwLockWriteGuard<T> {
            self.0.try_write().unwrap()
        }
    }
}

#[cfg(feature = "parking_lot")]
pub use self::parking_lot::{Mutex, RwLock};

#[cfg(not(feature = "parking_lot"))]
pub use std::{Mutex, RwLock};
