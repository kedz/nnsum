import pathlib

from ignite.exceptions import NotComputableError
from ignite.metrics.metric import Metric

import rouge_papier


class PerlRouge(Metric):
    """
    Calculates the average rouge score using the original perl rouge script.
    """
    def __init__(self, summary_length, remove_stopwords=True, 
                 delete_temp_files=True, output_transform=lambda x: x):
        super(PerlRouge, self).__init__(output_transform)
        self._summary_length = summary_length
        self._remove_stopwords = remove_stopwords
        self._delete_temp_files = delete_temp_files

    @property
    def summary_length(self):
        return self._summary_length

    @property
    def remove_stopwords(self):
        return self._remove_stopwords

    @property
    def delete_temp_files(self):
        return self._delete_temp_files

    def reset(self):
        self._path_data = []

    def update(self, output):
        self._path_data.extend(output)

    def compute(self):
        if len(self._path_data) == 0:
            raise NotComputableError(
                'PerlRouge must have at least one example before ' \
                'it can be computed')

        with rouge_papier.util.TempFileManager() as manager:

            config_text = rouge_papier.util.make_simple_config_text(
                self._path_data)
            config_path = manager.create_temp_file(config_text)
            df = rouge_papier.compute_rouge(
                config_path, max_ngram=2, lcs=False, 
                remove_stopwords=self.remove_stopwords,
                length=self.summary_length)

        if self.delete_temp_files:
            for paths in self._path_data:
                pathlib.Path(paths[0]).unlink()  

        return df.iloc[-1:].to_dict("records")[0]
