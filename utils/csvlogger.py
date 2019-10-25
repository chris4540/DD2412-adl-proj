import csv
import collections

class CustomizedCSVLogger:
    """
    Reference:
    tf.keras.callbacks.CSVLogger
    """

    def __init__(self, filename, sep=',', append=False):
        self.filename = filename
        self.sep = sep
        self.headers = None
        # self.append = append
        # self._recs = None
        self.__header_written = append


    def log(self, **kwargs):

        self.row_dict = collections.OrderedDict(kwargs)

        if not self.headers:
            self.headers = list(self.row_dict.keys())
        self.__write()

    def __write(self):
        if self.__header_written:
            mode = 'a'
        else:
            mode = 'w'

        with open(self.filename, mode) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.headers)

            if not self.__header_written:
                writer.writeheader()
                self.__header_written = True

            writer.writerow(self.row_dict)

if __name__ == "__main__":
    logger = CustomizedCSVLogger('test.csv')
    logger.log(epoch=1, err=2)
    logger.log(epoch=2, err=0.2)

