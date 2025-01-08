use std::rc::Rc;

use crate::caching::bounded::list_node::SharedNode;

pub struct DoublyLinkedList<K, V> {
    head: Option<SharedNode<K, V>>,
    tail: Option<SharedNode<K, V>>,
}

impl<K, V> DoublyLinkedList<K, V> {
    pub(crate) fn new() -> Self {
        Self {
            head: None,
            tail: None,
        }
    }

    pub(crate) fn add_to_head(&mut self, node: SharedNode<K, V>) {
        node.borrow_mut().next = self.head.clone();
        node.borrow_mut().prev = None;

        if let Some(head) = self.head.clone() {
            head.borrow_mut().prev = Some(Rc::downgrade(&node));
        }

        self.head = Some(node.clone());

        if self.tail.is_none() {
            self.tail = Some(node);
        }
    }

    pub(crate) fn remove(&mut self, node: SharedNode<K, V>) {
        let prev = node.borrow().prev.clone();
        let next = node.borrow().next.clone();

        if let Some(prev_node) = prev.as_ref().and_then(|weak| weak.upgrade()) {
            prev_node.borrow_mut().next = next.clone();
        } else {
            self.head = next.clone();
        }

        if let Some(next_node) = next {
            next_node.borrow_mut().prev = prev;
        } else {
            self.tail = prev.and_then(|weak| weak.upgrade());
        }
    }

    pub(crate) fn remove_tail(&mut self) -> Option<SharedNode<K, V>> {
        if let Some(tail) = self.tail.clone() {
            self.remove(tail.clone());
            Some(tail)
        } else {
            None
        }
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::caching::bounded::list_node::Node;

    #[test]
    fn test_add_to_head() {
        let mut list = DoublyLinkedList::new();
        let node1 = Node::new(1, 10);
        let node2 = Node::new(2, 20);

        list.add_to_head(node1.clone());
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 1);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 1);

        list.add_to_head(node2.clone());
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 2);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 1);
    }

    #[test]
    fn test_remove_node() {
        let mut list = DoublyLinkedList::new();
        let node1 = Node::new(1, 10);
        let node2 = Node::new(2, 20);
        let node3 = Node::new(3, 30);

        list.add_to_head(node1.clone());
        list.add_to_head(node2.clone());
        list.add_to_head(node3.clone());

        // List is now: {3, 2, 1}
        list.remove(node2.clone());
        // List should now be: {3, 1}
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 3);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 1);

        list.remove(node3.clone());
        // List should now be: {1}
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 1);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 1);

        list.remove(node1.clone());
        // List should now be empty
        assert!(list.head.is_none());
        assert!(list.tail.is_none());
    }

    #[test]
    fn test_remove_tail() {
        let mut list = DoublyLinkedList::new();
        let node1 = Node::new(1, 10);
        let node2 = Node::new(2, 20);

        list.add_to_head(node1.clone());
        list.add_to_head(node2.clone());

        // List is now: {2, 1}
        let removed_tail = list.remove_tail().unwrap();
        assert_eq!(removed_tail.borrow().key, 1);
        // List should now be: {2}
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 2);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 2);

        let removed_tail = list.remove_tail().unwrap();
        assert_eq!(removed_tail.borrow().key, 2);
        // List should now be empty
        assert!(list.head.is_none());
        assert!(list.tail.is_none());
    }

    #[test]
    fn test_add_and_remove_combination() {
        let mut list = DoublyLinkedList::new();
        let node1 = Node::new(1, 10);
        let node2 = Node::new(2, 20);
        let node3 = Node::new(3, 30);

        list.add_to_head(node1.clone());
        list.add_to_head(node2.clone());
        list.add_to_head(node3.clone());

        // List is now: {3, 2, 1}
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 3);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 1);

        list.remove(node1.clone());
        // List should now be: {3, 2}
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 3);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 2);

        list.add_to_head(node1.clone());
        // List should now be: {1, 3, 2}
        assert_eq!(list.head.as_ref().unwrap().borrow().key, 1);
        assert_eq!(list.tail.as_ref().unwrap().borrow().key, 2);
    }
}
