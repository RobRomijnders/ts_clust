from setuptools import setup, find_packages

setup(name='ts_clust',
      version='0.1',
      description='',
      url='',
      author='Rob_Romijnders, Ivan Sekulic',
      author_email='romijndersrob@gmail.com',
      license='MIT_license',
      install_requires=[
          'numpy',
          'matplotlib',
          'scipy'
      ],
      packages=find_packages(exclude=('tests')),
      zip_safe=False)
