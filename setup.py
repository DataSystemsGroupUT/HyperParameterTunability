import setuptools
with open("README.md", "r", encoding="utf-8") as f:
      long_description = f.read()

setuptools.setup(name='hyperparameter_tunability',
      version='0.1',
      author = "Javad Bahmani, Radwa Elshawi",
      author_email = "mbahmani@ut.ee",
      description='Hyperparameter importance for classification problems.',
      long_description=long_description,
      license = "MIT",
      url = "https://github.com/DataSystemsGroupUT/HyperParameterTunability",
      package_dir={"": "hyperparameter_tunability"},
      packages=["hyperparameter_tunability"])
