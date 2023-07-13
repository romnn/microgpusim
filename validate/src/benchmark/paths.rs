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
            base.as_ref().join(self).normalize()
        }
    }

    #[inline]
    #[must_use]
    fn relative_to<P>(&self, base: P) -> PathBuf
    where
        P: AsRef<Path>,
    {
        let rel_path: PathBuf =
            pathdiff::diff_paths(self, base).unwrap_or_else(|| self.to_path_buf());
        rel_path.normalize()
    }

    #[must_use]
    fn try_normalize(&self) -> Result<PathBuf, std::io::Error> {
        Ok(utils::fs::normalize_path(self))
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

    #[test]
    fn test_path_resolve_on_absolute_path() -> eyre::Result<()> {
        let absolute_path = PathBuf::from("/base/vectoradd/vectoradd");
        diff_assert_eq!(absolute_path.resolve("/another-base"), absolute_path,);
        diff_assert_eq!(absolute_path.resolve("test"), absolute_path,);
        diff_assert_eq!(absolute_path.resolve("../something"), absolute_path,);
        diff_assert_eq!(absolute_path.resolve(""), absolute_path,);
        Ok(())
    }

    #[test]
    fn test_path_resolve_absolute_base() -> eyre::Result<()> {
        diff_assert_eq!(
            PathBuf::from("./vectoradd/vectoradd").resolve("/base/"),
            PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff_assert_eq!(
            PathBuf::from("././vectoradd/vectoradd").resolve("/base"),
            PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff_assert_eq!(
            PathBuf::from("vectoradd/vectoradd").resolve("/base"),
            PathBuf::from("/base/vectoradd/vectoradd")
        );
        diff_assert_eq!(
            PathBuf::from("vectoradd/vectoradd")
                .resolve("/base")
                .resolve("/base"),
            PathBuf::from("/base/vectoradd/vectoradd")
        );
        Ok(())
    }

    #[test]
    fn test_path_resolve_relative_base() -> eyre::Result<()> {
        diff_assert_eq!(
            PathBuf::from("./vectoradd/vectoradd").resolve("base/"),
            PathBuf::from("base/vectoradd/vectoradd")
        );
        diff_assert_eq!(
            PathBuf::from("././vectoradd/vectoradd").resolve("./base/test/"),
            PathBuf::from("base/test/vectoradd/vectoradd")
        );
        // at the moment, we do not guard against possibly unwanted behaviour when resolving
        // multiple times on the same relative path accidentally.
        diff_assert_eq!(
            PathBuf::from("vectoradd/vectoradd")
                .resolve("base")
                .resolve("base"),
            PathBuf::from("base/base/vectoradd/vectoradd")
        );
        Ok(())
    }
}
