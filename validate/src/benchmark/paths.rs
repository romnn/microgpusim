use std::path::{Path, PathBuf};
pub trait PathExt {
    #[must_use]
    fn resolve<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>;

    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>;

    #[must_use]
    fn normalize(&self) -> PathBuf;

    #[must_use]
    fn try_normalize(&self) -> Result<PathBuf, std::io::Error>;
}

impl PathExt for Path {
    #[inline]
    #[must_use]
    fn resolve<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        if self.is_absolute() {
            self.normalize()
        } else {
            base.as_ref().join(&self).normalize()
        }
    }

    #[inline]
    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        let rel_path: PathBuf =
            pathdiff::diff_paths(&self, base).unwrap_or_else(|| self.to_path_buf());
        rel_path.normalize()
    }

    #[must_use]
    fn try_normalize(&self) -> Result<PathBuf, std::io::Error> {
        /// source:
        /// https://github.com/rust-lang/cargo/blob/fede83ccf973457de319ba6fa0e36ead454d2e20/src/cargo/util/paths.rs#L61
        use std::path::Component;
        let mut components = self.components().peekable();
        let mut ret = if let Some(c @ Component::Prefix(..)) = components.peek().cloned() {
            components.next();
            PathBuf::from(c.as_os_str())
        } else {
            PathBuf::new()
        };

        for component in components {
            match component {
                Component::Prefix(..) => unreachable!(),
                Component::RootDir => {
                    ret.push(component.as_os_str());
                }
                Component::CurDir => {}
                Component::ParentDir => {
                    ret.pop();
                }
                Component::Normal(c) => {
                    ret.push(c);
                }
            }
        }
        Ok(ret)
    }

    #[inline]
    #[must_use]
    fn normalize(&self) -> PathBuf {
        self.try_normalize().unwrap_or_else(|_| self.to_path_buf())
    }
}

#[cfg(test)]
mod tests {
    use super::PathExt;
    use color_eyre::eyre;
    use pretty_assertions::assert_eq as diff_assert_eq;
    use std::path::PathBuf;

    #[test]
    fn test_path_normalize() -> eyre::Result<()> {
        diff_assert_eq!(
            PathBuf::from("/base/./vectoradd/vectoradd").try_normalize()?,
            PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff_assert_eq!(
            PathBuf::from("/base/../vectoradd/vectoradd").try_normalize()?,
            PathBuf::from("/vectoradd/vectoradd")
        );
        Ok(())
    }
}
