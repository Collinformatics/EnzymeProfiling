class TrieNode:
    # Create nodes
    def __init__(self):
        self.children = {}
        self.endOfWord = False


class Trie:
    def __init__(self):
        # Create the root as an empty node
        self.root = TrieNode()

    def insert(self, word: str) -> None:
        # Add words to the trie
        current =  self.root

        for AA in word:
            if AA not in current.children:
                current.children[AA] = TrieNode()
            current = current.children[AA]

    def search(self, word):
        # Search for a word in the trie
        current = self.root

        for AA in word:
            if AA not in current.children:
                return False
            current = current.children[AA]
        return True

    def startsWith(self, prefix):
        # Find the prefix in the trie
        current = self.root

        for AA in prefix:
            if AA not in current.children:
                return False
        return True
