Code contribution
=================

Steps for submitting your code
------------------------------

When contributing code follow this checklist:

    #. Fork the repository on GitHub.
    #. Create an issue with the desired feature or bug fix.
    #. Make your modifications or additions in a feature branch.
    #. Make changes and commit your changes using a descriptive commit message.
    #. Provide tests for your changes, and ensure they all pass.
    #. Provide documentation for your changes, in accordance with the style of the rest of the project (see :ref:`style_guide`).
    #. Create a pull request to RandomFields main branch. The STEM team will review and discuss your Pull Request with you.

For any questions, please get in contact with one of the members of :doc:`authors`.


.. _style_guide:

Code style guide
----------------
The additional features should follow the style of the RandomFields project.

We follow the PEP 8 style guide for Python code, with our custom modifications as defined in the
`Yapf file <../../.style.yapf>`_ and the `flake8 file <../../.flake8>`_. These files can be ran manually by using the
following command from the root directory of the project:

.. code-block::

    pre-commit run --all-files


The class or function name should be clear and descriptive of the functionality it provides.

There should be a docstring at the beginning of the class or function describing its purpose and usage.
The docstring should be in the form of a triple-quoted string.

The class or function must have a type annotation.
The class should specify the attributes and inheritance.
The function should specify the arguments, exceptions and returns (in this order).
The return type annotation should be in the form of a comment after the closing parenthesis of the arguments.

Please, avoid inheritance, and favour composition when writing your code.

An example of a class:

.. code-block::

    class ClassName(object):
        """
        Work in progress: docstring style to be specified

        Inheritance:
            -:class:`object`

        Attributes:
            - <attribute_name_1> (<attribute type>): <attribute description>,shape (n,m)
            - <attribute_name_2> (<attribute type>): <attribute description>,shape (m,)


        """
        def __init__(self,<args>,...):
            """
            Constructor of the RandomFields class

            Args:
                - <arg>: (<type>) <description>, shape (n,m,), default
            """
            self.attribute_name_1: <attribute_type> = <value>
            self.attribute_name_2: <attribute_type> = <value>


An example of a function:

.. code-block::

    def generate(self, nodes: npt.NDArray[np.float64]) -> None:
        """
        Generate random field

        Args:
            - nodes (ndarray): The nodes of the random field. shape (:,`self.n_dim`)

        Raises:
            - ValueError: if dimensions of `nodes` do not match dimensions of the model

        Returns:

        """

        # check dimensions of nodes agrees with dimensions of model
        if nodes.shape[1] != self.n_dim:
            raise ValueError(f'Dimensions of nodes: {nodes.shape[1]} do not match dimensions of model: {self.n_dim}')

        # scale of fluctuation
        scale_fluctuation = np.ones(self.n_dim) * self.vertical_scale_fluctuation

        # apply the anisotropy to the other dimensions
        mask = np.arange(len(scale_fluctuation)) != self.v_dim
        scale_fluctuation[mask] = scale_fluctuation[mask] * self.anisotropy

        model = self.random_field_model(dim = self.n_dim,
                                        var = self.variance,
                                        len_scale = scale_fluctuation,
                                        angles = self.angle)
        self.random_field = gs.SRF(model,
                                   mean = self.mean,
                                   seed = self.seed)
        self.random_field(nodes.T)
