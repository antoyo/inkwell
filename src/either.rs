#[derive(Debug)]
pub enum Either<S, T> {
    Left(S),
    Right(T),
}

pub use Either::{Left, Right};

impl<S, T> Either<S, T> {
    pub fn is_right(&self) -> bool {
        match self {
            Left(_) => false,
            Right(_) => true,
        }
    }

    pub fn left(self) -> Option<S> {
        match self {
            Left(value) => Some(value),
            Right(_) => None,
        }
    }

    pub fn right(self) -> Option<T> {
        match self {
            Left(_) => None,
            Right(value) => Some(value),
        }
    }
}
