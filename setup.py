from setuptools import setup, find_packages


setup(
   name='nnsum',
   version='1.0',
   description='A neural network based extractive summarization library.',
   author='Chris Kedzie',
   author_email='kedzie@cs.columbia.edu',
   packages=find_packages(),
   dependency_links = [
       'git+https://github.com/kedz/rouge_papier.git#egg=rouge_papier'],
   install_requires = ["rouge_papier", "pytorch-ignite", "ujson", "colorama"],
)
