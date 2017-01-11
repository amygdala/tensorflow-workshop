from setuptools import setup

setup(name='widendeep',
      packages=['widendeep'],
      version='0.1',
      description='TensorFlow Wide and Deep example',
      url='https://github.com/amygdala/tensorflow-workshop',
      author='Yufeng Guo',
      author_email='yfg@google.com',
      license='MIT',
      install_requires=['tensorflow==0.12.1'],
      #dependency_links=['https://storage.googleapis.com/tensorflow/linux/cpu/tensorflow-0.12.1-cp27-none-linux_x86_64.whl'],
      zip_safe=False)
