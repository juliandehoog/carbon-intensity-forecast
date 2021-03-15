from setuptools import setup

setup(name="ciforecast",
      version="0.1.1",
      description="Basic Pip Module for Carbon Intensity Forecasting",
      url="https://github.com/juliandehoog/carbon-intensity-forecast",
      author="Julian de Hoog",
      author_email="julian@dehoog.ca",
      packages=[
          "ciforecast",
      ],
      install_requires=[
          "pandas",
          "sktime",
          "numpy",
          "pytz",
      ],
      tests_require=[],
      zip_safe=False)
