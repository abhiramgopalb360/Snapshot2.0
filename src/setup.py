from setuptools import setup, find_packages

setup(name='vyper',
      version='2.0.3',
      description='VIPER 2.0',
      url='https://github.com/charlesdublend360/viper_2',
      author='Blend360',
      package_dir={'': 'src'},
      python_requires='>=3.5',
      packages=find_packages(where='src'),
      include_package_data=True,
      install_requires=[
            'pandas>=1.1.0, <=1.2.3',
            'statsmodels>=0.11.1, <=0.12.2',
            'scikit-learn>=0.23.0, <=0.24.1',
            'scipy>=1.5.0, <=1.6.2',
            'openpyxl>=2.6.2, <=3.0.7'
      ],
      zip_safe=False)
