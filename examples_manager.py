import os
import json
from typing import Dict, Any, List

class ExamplesManager:
    # Map example types to their corresponding files
    EXAMPLE_FILES = {
        "query_understanding": "query_understanding.json"  # Example queries for query understanding
    }

    def __init__(self):
        self.examples_dir = os.path.dirname(os.path.abspath(__file__))
        self.queries_dir = os.path.join(self.examples_dir, "examples")
        self._cache: Dict[str, List[Dict[str, Any]]] = {}

    def get_available_examples(self) -> list[str]:
        """Returns a list of available example types for LLM queries"""
        return list(self.EXAMPLE_FILES.keys())

    def load_example(self, example_type: str) -> List[Dict[str, Any]]:
        """
        Load a specific example query set by its type.
        Args:
            example_type: Type of example queries to load (e.g., 'query_understanding')
        Returns:
            List of dictionaries containing the example queries and their structured responses
        Raises:
            ValueError: If the example type is not found
        """
        if example_type in self._cache:
            return self._cache[example_type]

        if example_type not in self.EXAMPLE_FILES:
            raise ValueError(f"Example type '{example_type}' not found. Available types: {', '.join(self.get_available_examples())}")

        file_path = os.path.join(self.queries_dir, self.EXAMPLE_FILES[example_type])
        if not os.path.exists(file_path):
            raise ValueError(f"Example file for type '{example_type}' not found at {file_path}")

        with open(file_path, 'r') as f:
            example_data = json.load(f)
            self._cache[example_type] = example_data
            return example_data

    def get_example(self, example_type: str) -> List[Dict[str, Any]]:
        """
        Get example queries and their responses by type.
        Args:
            example_type: Type of example to load (e.g., 'query_understanding')
        Returns:
            List of dictionaries containing the example queries and their structured responses
        """
        return self.load_example(example_type) 