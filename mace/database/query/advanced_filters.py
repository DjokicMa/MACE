"""
Advanced Filtering with SQL-like Syntax
=======================================
Provides advanced query capabilities with parentheses, LIKE operator, and more.
"""

import re
from typing import List, Dict, Any, Tuple, Optional, Union
from .filters import PropertyFilter
import operator


class AdvancedFilterParser:
    """Parses and executes advanced filter expressions with SQL-like syntax."""
    
    # Supported operators
    OPERATORS = {
        '>': operator.gt,
        '>=': operator.ge,
        '<': operator.lt,
        '<=': operator.le,
        '==': operator.eq,
        '=': operator.eq,  # SQL-style equals
        '!=': operator.ne,
        '<>': operator.ne,  # SQL-style not equals
        'LIKE': 'like',
        'NOT LIKE': 'not_like',
        'IN': 'in',
        'NOT IN': 'not_in',
        'IS': 'is',
        'IS NOT': 'is_not',
    }
    
    def __init__(self):
        self.tokens = []
        self.position = 0
        
    def parse(self, expression: str) -> Dict[str, Any]:
        """
        Parse a complex filter expression into an abstract syntax tree.
        
        Supports:
        - Basic comparisons: property > value
        - Logical operators: AND, OR
        - Parentheses: (expr1 AND expr2) OR expr3
        - LIKE operator: formula LIKE 'C%'
        - IN operator: space_group IN (225, 227, 229)
        - IS NULL/IS NOT NULL
        
        Args:
            expression: Filter expression string
            
        Returns:
            AST representation of the expression
        """
        self.tokens = self._tokenize(expression)
        self.position = 0
        
        if not self.tokens:
            return None
            
        return self._parse_or()
        
    def _tokenize(self, expression: str) -> List[Dict[str, Any]]:
        """Tokenize the expression into operators, identifiers, and values."""
        # Pattern to match various tokens
        patterns = [
            # Operators (order matters for multi-char operators)
            (r'IS\s+NOT\s+NULL', 'OPERATOR', 'IS NOT NULL'),
            (r'IS\s+NULL', 'OPERATOR', 'IS NULL'),
            (r'NOT\s+LIKE', 'OPERATOR', 'NOT LIKE'),
            (r'NOT\s+IN', 'OPERATOR', 'NOT IN'),
            (r'IS\s+NOT', 'OPERATOR', 'IS NOT'),
            (r'LIKE', 'OPERATOR', 'LIKE'),
            (r'IN', 'OPERATOR', 'IN'),
            (r'IS', 'OPERATOR', 'IS'),
            (r'>=', 'OPERATOR', '>='),
            (r'<=', 'OPERATOR', '<='),
            (r'!=', 'OPERATOR', '!='),
            (r'<>', 'OPERATOR', '<>'),
            (r'==', 'OPERATOR', '=='),
            (r'>', 'OPERATOR', '>'),
            (r'<', 'OPERATOR', '<'),
            (r'=', 'OPERATOR', '='),
            # Logical operators (with word boundaries)
            (r'\bAND\b', 'LOGICAL', 'AND'),
            (r'\bOR\b', 'LOGICAL', 'OR'),
            # Parentheses
            (r'\(', 'PAREN', '('),
            (r'\)', 'PAREN', ')'),
            # Values
            (r"'([^']*)'", 'STRING', None),  # Single quoted strings
            (r'"([^"]*)"', 'STRING', None),  # Double quoted strings
            (r'-?\d+\.\d+', 'NUMBER', None),  # Floats
            (r'-?\d+', 'NUMBER', None),      # Integers
            (r'NULL', 'NULL', None),         # NULL value
            (r'True|true|TRUE', 'BOOLEAN', True),
            (r'False|false|FALSE', 'BOOLEAN', False),
            # Identifiers (property names)
            (r'[a-zA-Z_][a-zA-Z0-9_]*', 'IDENTIFIER', None),
        ]
        
        tokens = []
        remaining = expression.strip()
        
        while remaining:
            matched = False
            
            # First check for list syntax if last token was IN/NOT IN
            if tokens and tokens[-1]['type'] == 'OPERATOR' and tokens[-1]['value'] in ('IN', 'NOT IN'):
                list_match = re.match(r'^\s*\(([^)]*)\)', remaining)
                if list_match:
                    # Parse list items
                    items_str = list_match.group(1)
                    items = []
                    for item in items_str.split(','):
                        item = item.strip()
                        if item.startswith("'") and item.endswith("'"):
                            items.append(item[1:-1])
                        elif item.startswith('"') and item.endswith('"'):
                            items.append(item[1:-1])
                        elif item:  # Check for non-empty item
                            try:
                                # Try to parse as number
                                if '.' in item:
                                    items.append(float(item))
                                else:
                                    items.append(int(item))
                            except ValueError:
                                # Keep as string
                                items.append(item)
                    
                    tokens.append({
                        'type': 'LIST',
                        'value': items
                    })
                    remaining = remaining[list_match.end():]
                    matched = True
            
            # If not matched as list, try regular patterns
            if not matched:
                for pattern, token_type, token_value in patterns:
                    regex = re.match(r'^\s*' + pattern, remaining, re.IGNORECASE)
                    if regex:
                        matched = True
                        
                        if token_type == 'STRING':
                            # Extract the string content
                            value = regex.group(1)
                        elif token_type == 'NUMBER':
                            value = float(regex.group(0)) if '.' in regex.group(0) else int(regex.group(0))
                        elif token_type == 'IDENTIFIER' and token_value is None:
                            value = regex.group(0).strip()
                        else:
                            value = token_value if token_value is not None else regex.group(0).upper()
                        
                        tokens.append({
                            'type': token_type,
                            'value': value
                        })
                        
                        remaining = remaining[regex.end():]
                        break
                        
            if not matched:
                raise ValueError(f"Unexpected token in expression: {remaining}")
                    
        return tokens
        
    def _parse_or(self) -> Dict[str, Any]:
        """Parse OR expressions (lowest precedence)."""
        left = self._parse_and()
        
        while self._peek() and self._peek()['type'] == 'LOGICAL' and self._peek()['value'] == 'OR':
            self._consume()  # consume OR
            right = self._parse_and()
            left = {
                'type': 'logical',
                'operator': 'OR',
                'left': left,
                'right': right
            }
            
        return left
        
    def _parse_and(self) -> Dict[str, Any]:
        """Parse AND expressions (higher precedence than OR)."""
        left = self._parse_comparison()
        
        while self._peek() and self._peek()['type'] == 'LOGICAL' and self._peek()['value'] == 'AND':
            self._consume()  # consume AND
            right = self._parse_comparison()
            left = {
                'type': 'logical',
                'operator': 'AND',
                'left': left,
                'right': right
            }
            
        return left
        
    def _parse_comparison(self) -> Dict[str, Any]:
        """Parse comparison expressions or parenthesized expressions."""
        # Check for opening parenthesis
        if self._peek() and self._peek()['type'] == 'PAREN' and self._peek()['value'] == '(':
            self._consume()  # consume (
            expr = self._parse_or()  # recursively parse inner expression
            
            if self._peek() and self._peek()['type'] == 'PAREN' and self._peek()['value'] == ')':
                self._consume()  # consume )
            else:
                raise ValueError("Missing closing parenthesis")
                
            return expr
            
        # Parse property name
        if not self._peek() or self._peek()['type'] != 'IDENTIFIER':
            raise ValueError("Expected property name")
            
        property_name = self._consume()['value']
        
        # Parse operator
        if not self._peek() or self._peek()['type'] != 'OPERATOR':
            raise ValueError(f"Expected operator after property '{property_name}'")
            
        operator_token = self._consume()
        operator_str = operator_token['value']
        
        # Handle special cases
        if operator_str == 'IS NULL':
            return {
                'type': 'comparison',
                'property': property_name,
                'operator': 'IS',
                'value': None
            }
        elif operator_str == 'IS NOT NULL':
            return {
                'type': 'comparison',
                'property': property_name,
                'operator': 'IS NOT',
                'value': None
            }
            
        # Parse value
        value_token = self._peek()
        if not value_token:
            raise ValueError(f"Expected value after operator '{operator_str}'")
            
        if operator_str in ('IN', 'NOT IN'):
            # Expect a list
            if value_token['type'] != 'LIST':
                raise ValueError(f"Expected list after {operator_str} operator")
            value = self._consume()['value']
        else:
            # Regular value
            if value_token['type'] not in ('STRING', 'NUMBER', 'BOOLEAN', 'NULL'):
                raise ValueError(f"Expected value after operator '{operator_str}'")
            value = self._consume()['value']
            
        return {
            'type': 'comparison',
            'property': property_name,
            'operator': operator_str,
            'value': value
        }
        
    def _peek(self) -> Optional[Dict[str, Any]]:
        """Peek at the current token without consuming it."""
        if self.position < len(self.tokens):
            return self.tokens[self.position]
        return None
        
    def _consume(self) -> Dict[str, Any]:
        """Consume and return the current token."""
        token = self.tokens[self.position]
        self.position += 1
        return token
        
    def evaluate(self, ast: Dict[str, Any], material: Dict[str, Any], 
                 properties: List[Dict[str, Any]]) -> bool:
        """
        Evaluate an AST against a material and its properties.
        
        Args:
            ast: Abstract syntax tree from parse()
            material: Material data dictionary
            properties: List of property dictionaries for the material
            
        Returns:
            True if the material matches the filter
        """
        if not ast:
            return True
            
        if ast['type'] == 'logical':
            left_result = self.evaluate(ast['left'], material, properties)
            right_result = self.evaluate(ast['right'], material, properties)
            
            if ast['operator'] == 'AND':
                return left_result and right_result
            elif ast['operator'] == 'OR':
                return left_result or right_result
                
        elif ast['type'] == 'comparison':
            return self._evaluate_comparison(ast, material, properties)
            
        return False
        
    def _evaluate_comparison(self, comparison: Dict[str, Any], 
                           material: Dict[str, Any], 
                           properties: List[Dict[str, Any]]) -> bool:
        """Evaluate a single comparison."""
        property_name = comparison['property']
        operator_str = comparison['operator']
        expected_value = comparison['value']
        
        # Get the actual value
        actual_value = self._get_property_value(property_name, material, properties)
        
        # Handle NULL checks
        if operator_str == 'IS':
            return actual_value is None if expected_value is None else actual_value == expected_value
        elif operator_str == 'IS NOT':
            return actual_value is not None if expected_value is None else actual_value != expected_value
            
        # If actual value is None and we're not doing NULL checks, return False
        if actual_value is None:
            return False
            
        # Handle LIKE operator
        if operator_str == 'LIKE':
            if not isinstance(actual_value, str):
                actual_value = str(actual_value)
            pattern = expected_value.replace('%', '.*').replace('_', '.')
            return bool(re.match(f'^{pattern}$', actual_value, re.IGNORECASE))
        elif operator_str == 'NOT LIKE':
            if not isinstance(actual_value, str):
                actual_value = str(actual_value)
            pattern = expected_value.replace('%', '.*').replace('_', '.')
            return not bool(re.match(f'^{pattern}$', actual_value, re.IGNORECASE))
            
        # Handle IN operator
        if operator_str == 'IN':
            return actual_value in expected_value
        elif operator_str == 'NOT IN':
            return actual_value not in expected_value
            
        # Handle regular operators
        if operator_str in self.OPERATORS:
            op_func = self.OPERATORS[operator_str]
            if callable(op_func):
                try:
                    # Type conversion for numeric comparisons
                    if isinstance(expected_value, (int, float)) and isinstance(actual_value, str):
                        try:
                            actual_value = float(actual_value)
                        except ValueError:
                            return False
                    return op_func(actual_value, expected_value)
                except (TypeError, ValueError):
                    return False
                    
        return False
        
    def _get_property_value(self, property_name: str, 
                          material: Dict[str, Any], 
                          properties: List[Dict[str, Any]]) -> Any:
        """Get a property value from material data or properties list."""
        # First check material attributes
        if property_name in material:
            value = material[property_name]
            # Convert numeric strings to numbers
            if isinstance(value, str) and value.replace('.', '').replace('-', '').isdigit():
                try:
                    return float(value) if '.' in value else int(value)
                except ValueError:
                    pass
            return value
            
        # Then check properties
        for prop in properties:
            if prop.get('property_name') == property_name:
                value = prop.get('property_value')
                # Try to convert to appropriate type
                if value is not None:
                    try:
                        # Try float first
                        return float(value)
                    except (ValueError, TypeError):
                        # Return as string
                        return str(value)
                return value
                
        return None


def parse_advanced_filter(expression: str) -> Dict[str, Any]:
    """
    Parse an advanced filter expression.
    
    Args:
        expression: Filter expression with SQL-like syntax
        
    Returns:
        AST representation of the expression
    """
    parser = AdvancedFilterParser()
    return parser.parse(expression)


def evaluate_advanced_filter(expression: str, material: Dict[str, Any], 
                           properties: List[Dict[str, Any]]) -> bool:
    """
    Evaluate an advanced filter expression against a material.
    
    Args:
        expression: Filter expression string
        material: Material data
        properties: Material properties
        
    Returns:
        True if material matches filter
    """
    parser = AdvancedFilterParser()
    ast = parser.parse(expression)
    return parser.evaluate(ast, material, properties)