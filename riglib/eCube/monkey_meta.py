import pickle
import pyforms

from PyQt5 import QtGui
from pyforms.basewidget import BaseWidget
from pyforms.controls import ControlFile
from pyforms.controls import ControlText
from pyforms.controls import ControlSlider
from pyforms.controls import ControlPlayer
from pyforms.controls import ControlButton
from pyforms.controls import ControlDir
from pyforms.controls import ControlList
from pyforms.controls import ControlDockWidget
from pyforms.controls import ControlEmptyWidget


# from AddMenuFuntionality import AddMenuFuntionality

# from pyforms.controls   import ControlDate
# from pyforms.controls   import ControlAutoComplete

class Entry(object):
    def __init__(self, monkeyname, exp, date, sessID, dir):
        self._monkeyname = monkeyname
        self._experimentor = exp
        self._date = date
        self._sessID = sessID
        self._dir = dir


class EntryWindow(Entry, BaseWidget):

    def __init__(self):
        Entry.__init__(self, '', '', '', '', '')
        BaseWidget.__init__(self, 'Entry Window')
        # super(EntryWindow, self).__init__('Monkey Meta data entry GUI')

        # Definition of the forms fields
        self._monkeynamefield = ControlText('Monkey', 'Default value')
        self._sessionidfield = ControlText('Session ID')
        self._datefield = ControlText('Date')
        self._experimentorfield = ControlText('Experimentor Name')
        self._filepathfield = ControlDir('Video Path')
        self._addentrybtnfield = ControlButton('Add Entry')

        self._addentrybtnfield.value = self.__addEntryAction

    def __addEntryAction(self):
        self._monkeyname = self._monkeynamefield.value
        self._sessID = self._sessionidfield.value
        self._date = self._datefield.value
        self._experimentor = self._experimentorfield.value
        self._dir = self._filepathfield.value

        if self.parent != None:
            self.parent.addEntry(self)


class MainList(object):

    def __init__(self):
        self._metadata = []

    def addEntry(self, entry):
        self._metadata.append(entry)
        self.save('metadata.dat')

    def removeEntry(self, index):
        return self._metadata.pop(index)
        #self.save('metadata.p')

    def save(self, filename):
        output = open(filename, 'wb')
        pickle.dump(self._metadata, output)

    def load(self, filename):
        pkl_file = open(filename, 'rb')
        self._metadata = pickle.load(pkl_file)


class AddMenuFuntionality(BaseWidget):
    """
    This class is a module of the application MainWindow
    This code adds the Open and Save functionality
    """

    def __init__(self):
        # It adds the next options to the main menu
        self.mainmenu = [
            {'File': [
                {'Save': self.__saveEntry},
                {'Open': self.__loadEntry},
                '-',
                {'Exit': self.__exit},
            ]
            }
        ]

    def __dummyEvent(self):
        exit()

    def __saveEntry(self):
        filename, _ = QtGui.QFileDialog.getSaveFileName(parent=self,
                                                        caption="Save file",
                                                        directory=".",
                                                        filter="*.dat")

        if filename != None and filename != '': self.save(filename)

    def __loadEntry(self):
        filename, _ = QtGui.QFileDialog.getOpenFileName(parent=self,
                                                        caption="Import file",
                                                        directory=".",
                                                        filter="*.dat")

        if filename != None and filename != '':
            self.load(filename)
            for entry in self._mainlist:
                self._mainlist += [entry._monkeyname, entry._experimentor,entry._date,entry._sessID, entry._dir ]

    def __exit(self):
        exit()


class MainListWindow(AddMenuFuntionality,MainList, BaseWidget):

    def __init__(self):
        MainList.__init__(self)
        BaseWidget.__init__(self, 'eCube videos - Meta Data')
        AddMenuFuntionality.__init__(self)

        # Defining Form Fields
        self._entrypanel = ControlDockWidget('Add Entry')
        self._mainlist = ControlList('Meta Data', readonly=True, add_function= self.__addEntryBtnAction(), remove_function=self.__rmEntryBtnAction())
        self._mainlist.horizontalHeaders = ['Monkey Name', 'Experimentor', 'Date', 'Session ID', 'File Dir']

        # m_list = MainList()
        # m_list.load('metadata.pkl')
        # print(m_list._metadata)
        #self._mainlist.value = super(MainListWindow,self).load('metadata.pkl')

        win = EntryWindow()
        win.parent = self
        self._entrypanel.value = win

    def __dummyEvent(self):
        print('dummy event')

    def closeEvent(self, event):
        print("called on close")

    def initForm(self):
        super(MainListWindow, self).initForm()

        self.mainmenu[0]['File'][0]['Save'].setEnabled(True)

    def __addEntryBtnAction(self):
        """
        Add entry button event
        """
        win = EntryWindow()
        win.parent = self
        self._entrypanel.value = win

    def __rmEntryBtnAction(self):
        """
        Remove entry button event
        """
        #super(MainListWindow, self).removeEntry(self._mainlist.selected_row_index)

    def addEntry(self, entry):
        """
        Update the GUI with the new entry
        """
        super(MainListWindow, self).addEntry(entry)
        self._mainlist += [entry._monkeyname, entry._experimentor, entry._date, entry._sessID, entry._dir]
        #super(MainListWindow,self).save('metadata.pkl')
        #entry.close()

    def removeEntry(self, index):
        super(MainListWindow, self).removeEntry(index)
        self._mainlist -= index


# Execute the application
if __name__ == "__main__":
    thislist = MainList()
    new_entry1 = Entry('Affi', 'Gus', '08-12-2020','1','<path>')
    thislist.addEntry(new_entry1)

    new_entry2 = Entry('Affi', 'Lydia', '08-12-2020', '2', '<path>')
    thislist.addEntry(new_entry2)
    thislist.save('metadata.dat')

    pyforms.start_app(MainListWindow)
