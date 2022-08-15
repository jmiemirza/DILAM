from os.path import exists
import pandas as pd
import logging

class ResultsManager():
    """
        Singleton class to manage results.
    """
    _instance = None
    log = logging.getLogger('MAIN.RESULTS')

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super(ResultsManager, cls).__new__(cls)
        return cls._instance


    def __init__(self):
        if hasattr(self, 'results'):
            return
        columns = ['method', 'task', 'value', 'scenario']
        self.results = pd.DataFrame(columns=columns)


    def save_to_file(self, file_name=None):
        if not file_name:
            path = 'results/raw_results_df.pkl'
        else:
            path = 'results/' + file_name
        self.results.to_pickle(path)


    def load_from_file(self, file_name=None):
        if not file_name:
            path = 'results/raw_results_df.pkl'
        else:
            path = 'results/' + file_name
        if not exists(path):
            raise Exception('Results file not found')
        self.results = pd.read_pickle(path)


    def add_result(self, method, task, error, scenario):
        entry = pd.DataFrame([{
            'method' : method,
            'task': task,
            'value': error,
            'scenario': scenario
        }])
        self.results = pd.concat([self.results, entry], ignore_index=True)


    def generate_summary(self):
        self.summary = {}
        tasks = self.results.task.unique()
        methods = self.results.method.unique()
        self.summary['online'] = pd.DataFrame(columns=tasks)
        self.summary['offline'] = pd.DataFrame(columns=tasks)

        for method in methods:
            for scenario in ['online', 'offline']:
                df = self.results[(self.results['method'] == method) &
                                  self.results['scenario'].isin([scenario, None])]
                if not len(df):
                    continue
                self.summary[scenario].loc[method] = list(df['value'])


    def print_summary(self):
        if not hasattr(self, 'summary'):
            self.generate_summary()
        self.log.info('Results summary:')
        pd.set_option('display.max_columns', None)
        for scenario, scenario_summary in self.summary.items():
            self.log.info(scenario.upper(), ':')
            self.log.info(scenario_summary, '\n')

    def print_summary_latex(self, max_cols=8):
        from math import ceil
        import warnings
        warnings.simplefilter(action='ignore', category=FutureWarning)

        if not hasattr(self, 'summary'):
            self.generate_summary()

        res = ('-' * 30) + 'START LATEX' +  ('-' * 30)
        for scenario in self.summary.keys():
            hdrs = self.summary[scenario].columns.values
            short_hdrs = [x.split('_')[0] for x in hdrs]
            length = len(hdrs)
            if max_cols == 0 or max_cols > length:
                max_cols = length
            start = 0
            end = min(max_cols, length)
            num_splits = ceil(length / max_cols)
            res += "\n\\begin{table}\n\\centering\n\\caption{" + scenario.capitalize() + "}\n"
            for x in range(num_splits):
                res += self.summary[scenario].to_latex(float_format="%.1f",
                                                       columns=hdrs[start:end],
                                                       header=short_hdrs[start:end])
                if x < num_splits - 1:
                    res += "\\vspace{-.6mm}\\\\\n"

                start += max_cols
                if x == num_splits-2:
                    end = length
                else:
                    end += max_cols

            res += "\\end{table}\n"

        res += ('-' * 30) + 'END LATEX' +  ('-' * 30)
        self.log.info(res)



    def plot_summary(self):
        import matplotlib.pyplot as plt
        import seaborn as sns
        import matplotlib.ticker as mticker

        sns.set_style("whitegrid")
        g = sns.FacetGrid(data=self.results, col='scenario', hue='method', legend_out=True)
        g.map(sns.lineplot, 'task', 'value', marker='o')
        g.add_legend()

        for axes in g.axes.flat:
            ticks_loc = axes.get_xticks()
            axes.xaxis.set_major_locator(mticker.FixedLocator(ticks_loc))
            axes.set_xticklabels(axes.get_xticklabels(), rotation=90)

            # shorten x axis labels by cutting anything after an underscore
            tasks_short = [x.get_text().split('_')[0] for x in axes.get_xticklabels()]
            axes.set_xticklabels(tasks_short)

            axes.tick_params(labelleft=True)
            axes.set_xlabel('Task')
            axes.set_ylabel('Error')

        plt.show(block=True)

    def plot_scenario_summary(self, scenario='online'):
        import matplotlib.pyplot as plt
        import seaborn as sns

        sns.set_style("whitegrid")
        df = self.results[self.results['scenario'] == scenario]
        lp = sns.lineplot(x='task', y='value', hue='method', data=df, marker='o')

        plt.legend(title='Method')
        sns.move_legend(lp, "upper left", bbox_to_anchor=(1, 1))
        plt.title(f'{scenario.capitalize()} scenario')
        plt.ylabel('Mean Error over current and previous tasks')
        plt.xlabel('Task')
        plt.xticks(rotation=90)
        plt.tight_layout()
        plt.show(block=True)

