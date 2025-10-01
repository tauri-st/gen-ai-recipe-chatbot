import unittest
import json
from unittest.mock import MagicMock
from gutenberg.recipes_storage_and_retrieval_v2 import (
    generate_nutrition_info_chain,
)

class TestRecipesStorageAndRetrieval(unittest.TestCase):

    ### ✅ Standard Tests ✅ ###

    def test_generate_nutrition_info_chain(self):
        mock_llm = MagicMock()
        mock_llm.return_value = '{"calories": 500, "protein": 30, "carbs": 50, "fat": 10}'

        chain = generate_nutrition_info_chain(mock_llm)
        result = chain.invoke({"text": "1 cup rice, 100g chicken, 1 tbsp olive oil"})

        mock_llm.assert_called()
        self.assertEqual(result, '{"calories": 500, "protein": 30, "carbs": 50, "fat": 10}')