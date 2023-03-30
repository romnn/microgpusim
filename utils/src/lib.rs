use std::ffi::CString;
use std::path::Path;

pub const GID_NOBODY: libc::gid_t = 65534;
pub const UID_NOBODY: libc::uid_t = 65534;

pub fn chown(
    path: impl AsRef<Path>,
    uid: libc::uid_t,
    gid: libc::gid_t,
    follow_symlinks: bool,
) -> std::io::Result<()> {
    let s = CString::new(path.as_ref().as_os_str().to_str().unwrap().as_bytes()).unwrap();
    let ret = unsafe {
        if follow_symlinks {
            libc::chown(s.as_ptr(), uid, gid)
        } else {
            libc::lchown(s.as_ptr(), uid, gid)
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
    uid: libc::uid_t,
    gid: libc::gid_t,
    follow_symlinks: bool,
) -> std::io::Result<()> {
    for entry in std::fs::read_dir(root.as_ref())? {
        let entry = entry?;
        let path = entry.path();
        chown(path, uid, gid, follow_symlinks)?;
    }
    Ok(())
}
