#![allow(clippy::missing_errors_doc, clippy::missing_panics_doc)]
pub mod build;

use std::path::Path;

pub const GID_NOBODY: libc::gid_t = 65534;
pub const UID_NOBODY: libc::uid_t = 65534;

pub fn chown(
    path: impl AsRef<Path>,
    user_id: libc::uid_t,
    group_id: libc::gid_t,
    follow_symlinks: bool,
) -> std::io::Result<()> {
    let s = std::ffi::CString::new(path.as_ref().as_os_str().to_str().unwrap().as_bytes()).unwrap();
    let ret = unsafe {
        if follow_symlinks {
            libc::chown(s.as_ptr(), user_id, group_id)
        } else {
            libc::lchown(s.as_ptr(), user_id, group_id)
        }
    };
    if ret == 0 {
        Ok(())
    } else {
        Err(std::io::Error::last_os_error())
    }
}

pub fn rchown(
    root: impl AsRef<Path>,
    user_id: libc::uid_t,
    group_id: libc::gid_t,
    follow_symlinks: bool,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root.as_ref())? {
        let entry = entry?;
        let path = entry.path();
        chown(path, user_id, group_id, follow_symlinks)?;
    }
    Ok(())
}
