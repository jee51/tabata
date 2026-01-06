from setuptools import setup

setup(
   name='tabata',
   version='1.0',
   description='Tabata est un package qui permet la manipulation de séries de signaux numériques.',
   author='Jérôme Lacaille',
   author_email='jerome.lacaille@gmail.com',
   packages=['tabata'],
   install_requires=[
      "numpy",
      "scipy",
      "pandas",
      "scikit-learn",
      "ipywidgets",
      "plotly",
      "matplotlib",
      "tables"
   ],
)
