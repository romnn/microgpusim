#[derive()]
#[repr(transparent)]
pub struct Owned<T: cxx::vector::VectorElement>(pub cxx::UniquePtr<cxx::Vector<T>>);

impl<'a, T> IntoIterator for &'a Owned<T>
where
    T: cxx::vector::VectorElement + 'a,
{
    type Item = &'a T;
    type IntoIter = cxx::vector::Iter<'a, T>;

    // #[inline]
    fn into_iter(self) -> Self::IntoIter {
        self.0.iter()
    }
}
