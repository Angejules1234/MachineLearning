import unittest
import pandas as pd
from pandas.testing import assert_frame_equal

from web import load_data, setup_reg, compare_models_reg, save_model_reg, pull_reg, setup_class, compare_models_class, save_model_class, pull_class
class TestDataLoader(unittest.TestCase):
    def setUp(self):
        self.test_file = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'target': [10, 20, 30]
        })
        self.expected_output = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'target': [10, 20, 30]
        })
    
    def test_load_data(self):
        # Arrange
        expected_output = self.expected_output
        
        # Act
        test_output = load_data(self.test_file)
        
        # Assert
        assert_frame_equal(test_output.reset_index(drop=True), expected_output.reset_index(drop=True))        
    
    def tearDown(self):
        # Clean up any resources used by the test
        pass


class TestRegression(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'target': [10, 20, 30]
        })
        self.expected_setup = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'target': [10, 20, 30]
        })
    
    
    def test_compare_models_reg(self):
        # Arrange
        expected_model_output_type = list
        
        # Act
        setup_reg(self.test_data, target='target')
        model_reg = compare_models_reg(sort="RMSE")
        
        # Assert
        assert isinstance(model_reg, expected_model_output_type)
    
    def tearDown(self):
        # Clean up any resources used by the test
        pass


class TestClassification(unittest.TestCase):
    def setUp(self):
        self.test_data = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })
        self.expected_setup = pd.DataFrame({
            'col1': [1, 2, 3],
            'col2': ['a', 'b', 'c'],
            'target': [0, 1, 0]
        })
    

        
    def test_compare_models_class(self):
        # Arrange
        expected_model_output_type = list
        
        # Act
        setup_class(self.test_data, target='target')
        model_class = compare_models_class(sort="AUC")
        
        # Assert
        assert isinstance(model_class, expected_model_output_type)
        
    def test_save_model_class(self):
        # Arrange
        expected_file_readable = True
        
        # Act
        setup_class(self.test_data, target='target')
        model_class=compare_models_class(sort="AUC")
        save_model_class(model_class, 'best_class_model')
        with open('best_class_model.pkl', 'rb') as f:
            file_readable = f.readable()
        
        # Assert
        assert file_readable == expected_file_readable


if __name__ == '__main__':
    unittest.main(argv=['first-arg-is-ignored'], exit=False)