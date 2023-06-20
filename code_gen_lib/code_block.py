
class CodeBlock:
    def __init__(self):
        pass

    def updateStatements(self, statements):
        assert (isinstance(statements, list)), "Provide all statements as a list of strings"
        assert any(isinstance(statement, (str,CodeBlock)) for statement in statements), "Expected statements to be in string format"
        self.statements = statements

"""
A code block containing statements
"""
class StatementBlock(CodeBlock):

    LINE_DELIMITER = "\n"

    def __init__(self, statements):
        super().__init__()
        assert (isinstance(statements, list)), "Provide all statements as a list of strings"
        assert any(isinstance(statement, (str,CodeBlock)) for statement in statements), "Expected statements to be in string format"
        self.statements = statements

    def __str__(self, indent=""):

        result = []

        for statement in self.statements:

            if isinstance(statement, CodeBlock):
                result.append(statement.__str__(indent))
            else:
                result.append(indent+statement+self.LINE_DELIMITER)

        return "".join(result)


"""
Condition Statement blocks
"""
class ConditionBlock(CodeBlock):

    LINE_DELIMITER = "\n"

    def __init__(self, condition, statements, block_type="if"):
        super().__init__()
        assert (isinstance(statements, list)), "Provide all statements as a list of strings"
        assert any(isinstance(statement, (str,CodeBlock)) for statement in statements), "Expected statements to be in string format"
        self.statements = statements

        assert isinstance(condition, str), "Expected condition to be a string statement"
        self.condition = condition

        assert (block_type.lower() in ["if", "elif", "else", "while", "for"]), f"Condition statements only contain if, elif, else, while and for blocks. Not {block_type}"
        self.block_type = block_type.lower()

    def getBlock(self, indent=""):

        if self.block_type in ["if", "elif", "while", "for"]:

            condition_wrap = f"({self.condition})" if self.block_type != "for" else self.condition

            return indent + f"{self.block_type} {condition_wrap}:" + self.LINE_DELIMITER
        else:

            return indent + "else:" + self.LINE_DELIMITER

    def __str__(self, indent=""):

        result = [self.getBlock(indent)]

        indent += "\t"

        for statement in self.statements:

            if isinstance(statement, CodeBlock):
                result.append(statement.__str__(indent))
            else:
                result.append(indent+statement+self.LINE_DELIMITER)

        return "".join(result)


"""
Function Block
"""

class FunctionBlock:

    LINE_DELIMITER = "\n"

    def __init__(self, func_name, parameters, statements):
        super().__init__()
        assert (isinstance(statements, list)), "Provide all statements as a list of strings"
        assert any(isinstance(statement, (str,CodeBlock)) for statement in statements), "Expected statements to be in string format"
        self.statements = statements

        assert isinstance(parameters, str), "Expected parameters to be a string statement"
        self.parameters = parameters

        assert isinstance(func_name, str), "Expected function name to be a string statement"
        self.func_name = func_name


    def getBlock(self, indent=""):

        return indent + f"def {self.func_name}({self.parameters}):" + self.LINE_DELIMITER

    def __str__(self, indent=""):

        result = [self.getBlock(indent)]

        indent += "\t"

        for statement in self.statements:

            if isinstance(statement, CodeBlock):
                result.append(statement.__str__(indent))
            else:
                result.append(indent+statement+self.LINE_DELIMITER)

        return "".join(result)