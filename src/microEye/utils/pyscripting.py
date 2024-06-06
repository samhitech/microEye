import sys
from subprocess import PIPE, Popen

from microEye.qt import (
    QT_API,
    QApplication,
    QtCore,
    QtGui,
    QtWidgets,
    Slot,
    getOpenFileName,
    getSaveFileName,
)

if 'PyQt' in QT_API:
    from microEye.qt import Qsci

    class CodeEditorWidget(Qsci.QsciScintilla):
        """
        Customized code editor widget based on QsciScintilla.

        Parameters
        ----------
        linter : str, optional
            The linter to be used (default is 'pyflakes').
        formatter : str, optional
            The code formatter to be used (default is 'autopep8').
        """

        def __init__(self, linter='pyflakes', formatter='autopep8'):
            """
            Initialize the CodeEditorWidget.

            Parameters
            ----------
            linter : str, optional
                The linter to be used (default is 'pyflakes').
            formatter : str, optional
                The code formatter to be used (default is 'autopep8').
            """
            super().__init__()

            self.setUtf8(True)
            self.setMarginType(0, Qsci.QsciScintilla.MarginType.NumberMargin)

            # 1. Text wrapping
            # -----------------
            self.setWrapMode(Qsci.QsciScintilla.WrapMode.WrapWord)
            self.setWrapVisualFlags(
                Qsci.QsciScintilla.WrapVisualFlag.WrapFlagByText,
                Qsci.QsciScintilla.WrapVisualFlag.WrapFlagInMargin,
            )
            self.setWrapIndentMode(Qsci.QsciScintilla.WrapIndentMode.WrapIndentSame)

            # 2. End-of-line mode
            # --------------------
            self.setEolMode(Qsci.QsciScintilla.EolMode.EolUnix)
            self.setEolVisibility(False)

            # 3. Indentation
            # ---------------
            self.setIndentationsUseTabs(False)
            self.setTabWidth(4)
            self.setIndentationGuides(True)
            self.setTabIndents(True)
            self.setAutoIndent(True)

            # 4. Caret
            # ---------
            # self.setCaretForegroundColor(QColor('#ff0000ff'))
            self.setCaretLineVisible(True)
            self.setCaretLineBackgroundColor(QtGui.QColor('#1f00007f'))
            self.setCaretForegroundColor(QtGui.QColor('#ffffff'))
            self.setCaretWidth(2)

            # 5. Margins
            # -----------
            # Margin 0 = Line nr margin
            self.setMarginType(0, Qsci.QsciScintilla.MarginType.NumberMargin)
            self.setMarginWidth(0, '0000000000')
            self.setMarginsForegroundColor(QtGui.QColor('#ffffff'))
            self.setMarginsBackgroundColor(QtGui.QColor('#181818'))
            # Margin 1 = Symbol
            # self.setMarginType(1, QsciScintilla.SymbolMarginColor)
            # self.setMarginWidth(0, '00')
            # self.markerDefine(QsciScintilla.Circle, 0)
            # self.setMarginMarkerMask(1, 0b1)

            self.linter = linter
            self.formatter = formatter

            self.__lexer = Qsci.QsciLexerPython(self)
            self.__lexer.setDefaultFont(QtGui.QFont('Consolas'))
            self.__lexer.setDefaultPaper(QtGui.QColor('#1F1F1F'))
            self.__lexer.setDefaultColor(QtGui.QColor('#f9f9f9'))
            self.__lexer.setColor(QtGui.QColor('#B5CEA8'), Qsci.QsciLexerPython.Number)
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.DoubleQuotedFString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.DoubleQuotedString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.SingleQuotedFString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.SingleQuotedString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.TripleDoubleQuotedFString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.TripleDoubleQuotedString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.TripleSingleQuotedFString
            )
            self.__lexer.setColor(
                QtGui.QColor('#CE9178'), Qsci.QsciLexerPython.TripleSingleQuotedString
            )

            self.__lexer.setColor(QtGui.QColor('#569CD6'), Qsci.QsciLexerPython.Keyword)
            self.__lexer.setColor(
                QtGui.QColor('#DCDCAA'), Qsci.QsciLexerPython.FunctionMethodName
            )
            self.__lexer.setColor(
                QtGui.QColor('#4DC9B0'), Qsci.QsciLexerPython.ClassName
            )
            self.__lexer.setColor(QtGui.QColor('#6A9955'), Qsci.QsciLexerPython.Comment)
            self.__lexer.setFont(QtGui.QFont('Consolas'), Qsci.QsciLexerPython.Comment)
            self.__lexer.setColor(
                QtGui.QColor('#6A9955'), Qsci.QsciLexerPython.CommentBlock
            )
            self.__lexer.setFont(
                QtGui.QFont('Consolas'), Qsci.QsciLexerPython.CommentBlock
            )
            self.__lexer.setColor(
                QtGui.QColor('#FFD70F'), Qsci.QsciLexerPython.Operator
            )

            self.__api = Qsci.QsciAPIs(self.__lexer)
            self.setLexer(self.__lexer)
            # self.setAutoCompletionSource(QsciScintilla.AcsAll)
            # self.setAutoCompletionThreshold(1)

            self.currentLineNumber = 0

            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.lintCode)
            self.timer.setSingleShot(True)
            self.timer.setInterval(100)

            self.textChanged.connect(self.debouncedLintCode)

        def debouncedLintCode(self):
            '''
            Start a timer to debounce code linting on text changes.
            '''
            self.timer.start()

        def lintCode(self):
            '''
            Lint the code using the specified linter.

            Returns
            -------
            str
                A message indicating the linting result.
            '''
            code = self.text()

            if self.linter == 'pyflakes':
                process = Popen(
                    ['pyflakes'], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True
                )
            else:
                return 'Unsupported linter.'

            output, errors = process.communicate(code)

            if errors:
                self.markLintingErrors(errors)
                return errors
            else:
                self.clearAnnotations()
                return 'Linting passed successfully.'

        def markLintingErrors(self, errors: str):
            '''
            Mark linting errors in the editor.

            Parameters
            ----------
            errors : str
                Error messages from the linter.
            '''
            error_lines_set = []
            error_lines = errors.splitlines()

            self.clearAnnotations()

            message = ''
            for line in error_lines:
                parts = line.split(':')
                if len(parts) >= 2 and parts[1].isdigit():
                    line_number = int(parts[1])
                    col_number = int(parts[2])
                    error_lines_set.append(line_number)
                    message += f'Col {col_number}: {parts[-1]}\n'
                else:
                    if len(line.strip()) > 0:
                        new_line = '' if '^' in line else '\n'
                        message += f'{line}{new_line}'

            self.annotate(line_number - 1, message, 3)

        def formatCode(self):
            '''
            Format the code using the specified code formatter.

            Returns
            -------
            str
                A message indicating the formatting result.
            '''
            code = self.text()

            if self.formatter == 'autopep8':
                process = Popen(
                    ['autopep8', '-'], stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True
                )
                formatted_code, errors = process.communicate(input=code)

                if errors:
                    return errors
                else:
                    self.setText(formatted_code)
                    return 'Code formatted successfully.'
            else:
                return 'Unsupported code formatter.'
else:
    class PythonHighlighter(QtGui.QSyntaxHighlighter):
        def __init__(self, parent=None):
            super().__init__(parent)
            self.highlightingRules = []

            keywordFormat = QtGui.QTextCharFormat()
            keywordFormat.setForeground(QtGui.QColor('#569CD6'))
            keywords = [
                'and', 'as', 'assert', 'break', 'class', 'continue','def', 'del',
                'elif', 'else', 'except', 'False', 'finally', 'for', 'from', 'global',
                'if', 'import', 'in', 'is', 'lambda', 'None', 'nonlocal', 'not',
                'or', 'pass', 'raise', 'return', 'True', 'try', 'while', 'with', 'yield'
                ]

            for keyword in keywords:
                pattern = QtCore.QRegularExpression(r'\b' + keyword + r'\b')
                self.highlightingRules.append((pattern, keywordFormat))

            # Add more patterns and formats for strings, comments, etc.

        def highlightBlock(self, text):
            for pattern, format in self.highlightingRules:
                expression = QtCore.QRegularExpression(pattern)
                index = expression.indexIn(text)
                while index >= 0:
                    length = expression.matchedLength()
                    self.setFormat(index, length, format)
                    index = expression.indexIn(text, index + length)

            self.setCurrentBlockState(0)

    class CodeEditorWidget(QtWidgets.QPlainTextEdit):
        def __init__(self, linter='pyflakes', formatter='autopep8', parent=None):
            super().__init__(parent)
            self.setFont(QtGui.QFont('Consolas', 10))
            self.setTabStopDistance(4 * self.fontMetrics().horizontalAdvance(' '))

            self.highlighter = PythonHighlighter(self.document())

            self.linter = linter
            self.formatter = formatter

            self.textChanged.connect(self.debouncedLintCode)
            self.timer = QtCore.QTimer(self)
            self.timer.timeout.connect(self.lintCode)
            self.timer.setSingleShot(True)
            self.timer.setInterval(100)

        def text(self, value):
            self.toPlainText(value)

        def setText(self, value):
            self.setPlainText(value)

        @Slot()
        def debouncedLintCode(self):
            self.timer.start()

        @Slot()
        def lintCode(self):
            code = self.toPlainText()
            if self.linter == 'pyflakes':
                process = Popen(
                    ['pyflakes', '-'],
                    stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
                output, errors = process.communicate(input=code)
                if errors:
                    self.markLintingErrors(errors)
                else:
                    self.clearAnnotations()

        def markLintingErrors(self, errors):
            # Add logic to mark linting errors
            pass

        def clearAnnotations(self):
            # Add logic to clear annotations
            pass

        @Slot()
        def formatCode(self):
            code = self.toPlainText()
            if self.formatter == 'autopep8':
                process = Popen(
                    ['autopep8', '-'],
                    stdin=PIPE, stdout=PIPE, stderr=PIPE, text=True)
                formatted_code, errors = process.communicate(input=code)
                if errors:
                    return errors
                else:
                    self.setPlainText(formatted_code)
                    return 'Code formatted successfully.'
            else:
                return 'Unsupported code formatter.'


class pyEditor(QtWidgets.QWidget):
    '''
    Main window containing the code editor.

    Parameters
    ----------
    parent : QWidget, optional
        The parent widget (default is None).
    '''

    def __init__(self, parent=None):
        '''
        Initialize the pyEditor.

        Parameters
        ----------
        parent : QWidget, optional
            The parent widget (default is None).
        '''
        super().__init__(parent=parent)

        self.mLayout = QtWidgets.QVBoxLayout()
        self.setLayout(self.mLayout)

        self.codeEditorWidget = CodeEditorWidget()
        self.codeEditorWidget.setAcceptDrops(True)
        self.codeEditorWidget.dragEnterEvent = self.pydragEnterEvent
        self.codeEditorWidget.dropEvent = (
            lambda e: self.codeEditorWidget.insertPlainText(e.mimeData().text())
        )

        self.btns_layout = QtWidgets.QHBoxLayout()
        self.open_btn = QtWidgets.QPushButton('Open', clicked=lambda: self.openScript())
        self.save_btn = QtWidgets.QPushButton('Save', clicked=lambda: self.saveScript())
        self.format_btn = QtWidgets.QPushButton(
            'Format', clicked=self.codeEditorWidget.formatCode
        )
        self.exec_btn = QtWidgets.QPushButton('Execute')
        self.btns_layout.addWidget(self.open_btn)
        self.btns_layout.addWidget(self.save_btn)
        self.btns_layout.addWidget(self.format_btn)
        self.btns_layout.addWidget(self.exec_btn)

        self.mLayout.addWidget(self.codeEditorWidget)
        self.mLayout.addLayout(self.btns_layout)

    def openScript(self):
        '''
        Open a Python script file and load its content into the editor.
        '''
        filename, _ = getOpenFileName(
            self, 'Load Script', filter='Python Files (*.py);;'
        )

        if len(filename) > 0:
            with open(filename, encoding='utf-8') as file:
                self.codeEditorWidget.setText(file.read())

    def saveScript(self):
        '''
        Save the content of the editor into a Python script file.
        '''
        filename, _ = getSaveFileName(
            self, 'Save Script', filter='Python Files (*.py);;'
        )

        if len(filename) > 0:
            with open(filename, 'w', encoding='utf-8') as file:
                file.write(self.codeEditorWidget.text())

    def pydragEnterEvent(self, e):
        '''
        Handle drag enter event for plain text.
        '''
        if e.mimeData().hasFormat('text/plain'):
            e.accept()
        else:
            e.ignore()

    def toPlainText(self):
        '''
        Get the plain text content of the code editor.

        Returns
        -------
        str
            The plain text content of the code editor.
        '''
        return self.codeEditorWidget.text()


if __name__ == '__main__':
    app = QApplication(sys.argv)
    window = pyEditor()
    window.show()
    sys.exit(app.exec())
