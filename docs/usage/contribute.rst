Contribute
=================================

We welcome contributions to DeLTA!

You can send us your training data if you think it will benefit the community.
Reach out via Gitlab's issue system :)

If you want to help with the development, you can fork DeLTA and submit Merge
Requests on Gitlab. If you plan on implementing significant changes, please
coordinate with us first.

Development environment
------------------------

We use a few of extra packages to facilitate development. These are listed
in the ``delta_dev.yml`` file, you can install it in the same way that the
``delta_env.yml`` file is :doc:`installed. <installation>`

Specifically the extra dependencies are:

* ``pytest``
* ``pytest-order``
* ``sphinx``
* ``numpydoc``
* ``pre-commit``

CI/CD pipeline
------------------

Thanks to gitlab's generous Open Source Program, we are now able to
systematically run tests for merge requests that merge into `dev` or `main`.
This is implemented via the `.gitlab-ci.yml` file at the root of the
systematically run tests for merge requests that merge into ``dev`` or ``main``.
This is implemented via the ``.gitlab-ci.yml`` file at the root of the
repository.

You can opt-out of the tests by adding the `tests::block` tag to the merge
request. Alternatively, you can force the tests to be trigger even if your
merge request does not merge into the `dev`/`main` branch by adding the
``tests::force`` tag to it. The ``tests::default`` is equivalent to not having
any tag.

Documentation
--------------

You can build the documentation locally with the following commands::

    $ sphinx-build -b html docs/ wherever/

However, we use `readthedocs.org <https://readthedocs.org/>`_ to automatically
build them and host them for us:

* The ``dev`` branch is monitored and publicly available under
  `delta.readthedocs.io/en/latest/ <https://delta.readthedocs.io/en/latest/>`_
* The ``main`` branch is monitored and publicly available under
  `delta.readthedocs.io/en/stable/ <https://delta.readthedocs.io/en/stable/>`_
* The ``docs`` branch is monitored, but is "hidden" and is meant for us to write
  or test the docs. It is available under
  `delta.readthedocs.io/en/stable/ <https://delta.readthedocs.io/en/stable/>`_
* Each release has its own "archive" and is available under its version number,
  e.g. `delta.readthedocs.io/en/2.0.0/ <https://delta.readthedocs.io/en/2.0.0/>`_

The docs on any other branch can be built on demand from the dashboard on
`readthedocs.org <https://readthedocs.org/dashboard>`_. To get access to it,
first create an account and one of the current maintainers needs to add you as
a maintainer under their admin menu. ``.readthedocs.yml`` manages settings
specific to the online build and deployment of the docs.
