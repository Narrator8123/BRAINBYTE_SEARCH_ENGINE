import json
import sys

from PyQt6.QtCore import QThread, pyqtSignal, QRect
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QIcon, QMovie, QPainter, QFont, QColor, QPixmap
from PyQt6.QtWidgets import (
    QApplication,
    QMainWindow,
    QPushButton,
    QLineEdit,
    QVBoxLayout,
    QWidget,
    QLabel,
    QScrollArea,
    QDialog,
    QMessageBox
)

from retrieval_model.predict import retrieve_function


# Load queries set from `sample_queries.tsv`
def load_queries_from_file(file_path):
    queries = set()
    with open(file_path, 'r') as file:
        for line in file:
            query = line.strip().split('\t')[1]
            queries.add(query)
    return queries


queries_set = load_queries_from_file('../data/sample_data/sample_queries.tsv')


# Wrapper function of retrieve_function() to format the results into JSON
def search_api(query):
    results, _, _ = retrieve_function(query)
    r, res_dic = {}, {}
    for idx, result in enumerate(results):
        contents = result.values()
        r = {}
        for content in contents:
            r["title"] = content[0]
            r["url"] = content[1]
            r["preview"] = content[2]
        res_dic[idx] = r
    return json.dumps(list(res_dic.values()))


class SearchThread(QThread):
    finished = pyqtSignal(str)  # Signal emitted with the search results in JSON format

    def __init__(self, query, parent=None):
        super(SearchThread, self).__init__(parent)
        self.query = query

    def run(self):
        # Perform the search operation
        results = search_api(self.query)
        self.finished.emit(results)  # Emit the results


# Waiting GIF shown while waiting for the results
class WaitingWindow(QDialog):
    def __init__(self, gif_path, parent=None):
        super(WaitingWindow, self).__init__(parent)
        self.setWindowFlags(Qt.WindowType.FramelessWindowHint | Qt.WindowType.WindowStaysOnTopHint)
        self.setModal(True)

        layout = QVBoxLayout()
        self.label = QLabel()
        self.movie = QMovie(gif_path)
        self.label.setMovie(self.movie)
        layout.addWidget(self.label)
        self.setLayout(layout)
        self.movie.start()
        self.setWindowIcon(QIcon('../UI/icon.png'))

    def closeWindow(self, event):
        self.movie.stop()
        super(WaitingWindow, self).closeWindow(event)


# Custom QLabel to display the watermark `Group24_ByteBrain` in the main window
class WatermarkLabel(QLabel):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setAttribute(Qt.WidgetAttribute.WA_TransparentForMouseEvents)

    def paintEvent(self, event):
        super().paintEvent(event)
        painter = QPainter(self)
        painter.setOpacity(0.2)
        painter.setFont(QFont('PT Mono', 15))
        painter.setPen(QColor(100, 100, 100))

        rect = self.rect()
        textHeight = 20
        textRect = QRect(rect.left(), rect.bottom() - textHeight - 10, rect.width(), textHeight)
        painter.drawText(textRect, Qt.AlignmentFlag.AlignCenter, '©ECS736P     Group24')
        painter.end()


# Searching window
class MainWindow(QMainWindow):
    def __init__(self):
        super().__init__()
        self.queries_set = queries_set
        self.initUI()
        self.waitingWindow = None
        self.searchThread = None
        self.watermarkLabel = WatermarkLabel(self)
        self.watermarkLabel.setGeometry(self.rect())
        self.resultsWindow = ResultsWindow(self)

    def initUI(self):
        self.setWindowTitle('Search Engine')
        self.setGeometry(1000, 400, 600, 150)

        # Set up the central widget and layout
        centralWidget = QWidget()
        self.setCentralWidget(centralWidget)
        self.mainLayout = QVBoxLayout()
        self.searchLayout = QVBoxLayout()

        # Create the logo
        self.logoLabel = QLabel()
        self.logoPixmap = QPixmap('../UI/qmul_logo.png')
        self.scaledLogo = self.logoPixmap.scaled(180, 90, Qt.AspectRatioMode.KeepAspectRatio)
        self.logoLabel.setPixmap(self.scaledLogo)
        self.logoLabel.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.mainLayout.addWidget(self.logoLabel)

        # Create the search input field
        self.searchLayout = QVBoxLayout()
        self.searchInput = QLineEdit()
        self.searchInput.setStyleSheet('QLineEdit {height: 30px;}')
        self.searchInput.setPlaceholderText('Please type your query:')
        self.searchInput.returnPressed.connect(self.onSearchClicked)  # Connect return press to search
        self.searchLayout.addWidget(self.searchInput)

        # Create the search button and connect it to the search function
        searchButton = QPushButton('Search')
        searchButton.clicked.connect(self.onSearchClicked)
        self.searchLayout.addWidget(searchButton)
        self.mainLayout.addLayout(self.searchLayout)

        centralWidget.setLayout(self.mainLayout)
        self.setWindowIcon(QIcon('../UI/icon.png'))

    # Adjust the size of watermark to match the main window
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.watermarkLabel.setGeometry(self.rect())

    def onSearchClicked(self):
        query = self.searchInput.text()
        if query not in self.queries_set:
            # Show a warning message if the query is invalid
            QMessageBox.warning(self, 'Invalid query', 'Query not valid. Please enter a query from the query set.')
            return

        # Hide research window while searching
        for i in range(self.searchLayout.count()):
            self.searchLayout.itemAt(i).widget().hide()
        self.hide()

        # Show the waiting dialog during searching process
        if not self.waitingWindow:
            self.waitingWindow = WaitingWindow('../UI/waiting_patiently.gif')
        self.waitingWindow.show()

        # Initialize the search thread / update its query
        query = self.searchInput.text()
        if self.searchThread is None:
            self.searchThread = SearchThread(query)
            self.searchThread.finished.connect(self.onSearchFinished)
        else:
            self.searchThread.query = query
        self.searchThread.start()

    def onSearchFinished(self, results):
        # Close the waiting dialog when the searching is finished
        if self.waitingWindow:
            self.waitingWindow.close()
            self.waitingWindow = None
        # Show the search window
        for i in range(self.searchLayout.count()):
            self.searchLayout.itemAt(i).widget().show()
        # Show the searching results
        self.resultsWindow.displayResults(self.searchInput.text(), lambda x: results)


class ResultsWindow(QMainWindow):
    def __init__(self, mainWindow):
        super().__init__()
        self.mainWindow = mainWindow
        self.initUI()

    def initUI(self):
        self.setWindowTitle('Results')
        self.setGeometry(600, 300, 600, 400)

        # Create the layout and scroll area to display results
        self.centralWidget = QWidget()
        self.layout = QVBoxLayout()
        self.scrollArea = QScrollArea()
        self.resultsContainer = QWidget()
        self.resultsLayout = QVBoxLayout(self.resultsContainer)
        self.scrollArea.setWidgetResizable(True)
        self.scrollArea.setWidget(self.resultsContainer)

        # Create the 'back' button to return to the main window
        self.backButton = QPushButton("Back")
        self.backButton.clicked.connect(self.onBackClicked)
        self.layout.addWidget(self.backButton)

        # Add the scroll area to the layout
        self.layout.addWidget(self.scrollArea)
        self.centralWidget.setLayout(self.layout)
        self.setCentralWidget(self.centralWidget)
        self.setWindowIcon(QIcon('../UI/icon.png'))

    def onBackClicked(self):
        self.hide()
        self.mainWindow.show()

    def displayResults(self, query, search_api):
        results_str = search_api(query)  # Call the search_api to get results for the query
        results = json.loads(results_str)
        results_filtered = {}

        # Clear previous search results
        for i in reversed(range(self.resultsLayout.count())):
            self.resultsLayout.itemAt(i).widget().setParent(None)

        # Filter the results to remove duplicate title
        for result in results:
            results_filtered[result['title']] = result

        # Create a QLabel for each search result, including a clickable link
        for r in results_filtered.values():
            last_space_position = r['preview'].rfind(' ')
            if last_space_position != -1:
                r['preview'] = r['preview'][:last_space_position] + '...'
            resultLabel = QLabel()
            resultLabel.setText(f"<div><h3><a href='{r['url']}'>{r['title']}</a></h3>"
                                f"<p>{r['preview']}</p></div>")
            resultLabel.setWordWrap(True)
            resultLabel.setOpenExternalLinks(True)
            self.resultsLayout.addWidget(resultLabel)

        self.resultsContainer.setLayout(self.resultsLayout)
        self.show()


def main():
    app = QApplication(sys.argv)
    mainWindow = MainWindow()
    mainWindow.show()
    sys.exit(app.exec())


if __name__ == '__main__':
    main()
