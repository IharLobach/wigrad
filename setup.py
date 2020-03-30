from distutils.core import setup
setup(
  name = 'wigrad',         # How you named your package folder (MyLib)
  packages = ['wigrad'],   # Chose the same as "name"
  version = '1.3',      # Start with a small number and increase it with every change you make
  license='MIT',        # Chose a license from here: https://help.github.com/articles/licensing-a-repository
  description = 'Caclulates the spectral-angular distribution of the number of photons emitted by one electron in a wiggler',   # Give a short description about your library
  author = 'Ihar Lobach',                   # Type in your name
  author_email = 'ilobach@uchicago.edu',      # Type in your E-Mail
  url = 'https://github.com/IharLobach/wigrad.git',   # Provide either the link to your github or to your website
  download_url = 'https://github.com/IharLobach/wigrad/archive/v1.3.tar.gz',    # I explain this later on
  keywords = ['wiggler', 'undulator', 'spectrum'],   # Keywords that define your package best
  install_requires=[            # I get to this in a second
          'numpy',
          'scipy',
          'matplotlib',
      ],
  classifiers=[
    'Development Status :: 3 - Alpha',      # Chose either "3 - Alpha", "4 - Beta" or "5 - Production/Stable" as the current state of your package
    'Intended Audience :: Developers',      # Define that your audience are developers
    'Topic :: Software Development :: Build Tools',
    'License :: OSI Approved :: MIT License',   # Again, pick a license
    'Programming Language :: Python :: 3',      #Specify which pyhton versions that you want to support
    'Programming Language :: Python :: 3.4',
    'Programming Language :: Python :: 3.5',
    'Programming Language :: Python :: 3.6',
    'Programming Language :: Python :: 3.7',
    'Programming Language :: Python :: 3.8'
  ],
)