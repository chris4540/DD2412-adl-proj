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
        self._header_written = append


    def log(self, **kwargs):

        row_dict = collections.OrderedDict(kwargs)

        self.log_with_order(row_dict)

    def log_with_order(self, ordered_dict):
        assert isinstance(ordered_dict, collections.OrderedDict)

        if not self.headers:
            self.headers = list(ordered_dict.keys())
        self._write(ordered_dict)

    def _write(self, row_dict):
        if self._header_written:
            mode = 'a'
        else:
            mode = 'w'

        with open(self.filename, mode) as csvfile:
            writer = csv.DictWriter(csvfile, fieldnames=self.headers)

            if not self._header_written:
                writer.writeheader()
                self._header_written = True

            writer.writerow(row_dict)

if __name__ == "__main__":
    logger = CustomizedCSVLogger('test.csv')
    logger.log(epoch=1, err=2)
    logger.log(epoch=2, err=0.2)

