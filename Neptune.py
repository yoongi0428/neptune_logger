from collections import OrderedDict
import neptune

class NeptuneLogger:
    def __init__(self, 
                api_key:str,
                project_name:str,
                experiment_name:str,
                description:str,
                tags:str,
                hparams:dict,
                upload_source_files:list=None,
                hostname:str='my-server',
                offline:bool=False):
        self.api_key = api_key
        self.project_name = project_name
        self.experiment_name = experiment_name
        self.description = description
        self.tags = tags
        self.hparams = hparams
        self.upload_source_files=upload_source_files
        self.hostname = hostname
        self.offline = offline
        
        self.initialize()

    @property
    def system_property(self):
        return self.experiment.get_system_properties()

    def initialize(self):
        # Get experiment
        if self.offline:
            project = neptune.Session(backend=neptune.OfflineBackend()).get_project('dry-run/project')
        else:
            session = neptune.Session.with_default_backend(api_token=self.api_key)
            project = session.get_project(self.project_name)
        
        exp = project.create_experiment(
            name=self.experiment_name,
            description=self.description,
            params=self.hparams,
            tags=self.tags,
            upload_source_files=self.upload_source_files,
            hostname=self.hostname)

        self.experiment = exp

    def log_hparams(self, hparams):
        for key, val in hparams.items():
            self.experiment.set_property(key, val)

    def log_metric(self, metric_name, value, epoch=None):
        if epoch is None:
            self.experiment.log_metric(metric_name, value)
        else:
            self.experiment.log_metric(metric_name, epoch, value)

    def log_metric_from_dict(self, metrics, epoch=None):
        for k, v in metrics.items():
            self.log_metric(k, v, epoch)

    def log_image(self, image_name, image, epoch=None):
        if epoch is None:
            self.experiment.log_image(image_name, image)
        else:
            self.experiment.log_image(image_name, epoch, image)

    def log_artifact(self, artifact, destination=None):
        self.experiment.log_artifact(artifact, destination)

if __name__ == '__main__':
    api='eyJhcGlfYWRkcmVzcyI6Imh0dHBzOi8vdWkubmVwdHVuZS5haSIsImFwaV91cmwiOiJodHRwczovL3VpLm5lcHR1bmUuYWkiLCJhcGlfa2V5IjoiNzZiMTkxYTctMjZmYS00NjlkLWE0ZDItNzQyNTkzNDhkYTc1In0='
    logger = NeptuneLogger(
        api_key=api,
        project_name='yoonkij/neptunetest',
        experiment_name='SanityCheckExp',
        description='Run for sanity check NeptuneLogger class',
        tags=['T1', 'T2'],
        hparams={'lr': 0.01, 'batch_size': 100, 'dimensions': [10, 50, 100]},
        upload_source_files=['README.md'],
        hostname='YK SERVER',
        offline=True
    )