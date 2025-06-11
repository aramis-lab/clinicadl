.. _api_introduction:

Introduction
============

The main new feature in ClinicaDL 2.0 is a Python API.
This API enables the user to manipulate various objects
coming from different libraries of the PyTorch ecosystem
for neuroimaging (mainly `PyTorch <https://pytorch.org/>`_,
`MONAI <https://monai.io/>`_ and
`TorchIO <https://torchio.readthedocs.io/>`_).

To account for this variability, ClinicaDL unifies
the manipulation of objects commonly used in Deep Learning
thanks to **configuration classes**.

Configuration classes
---------------------

In ClinicaDL, a configuration class is a dataclass that stores the
parameters of a Python object. For example, let's take the transform
:py:class:`torchio.transforms.Pad`:

.. code-block:: python

    >>> import torchio
    >>> torchio.Pad(padding=1)
    Pad(padding=1, padding_mode=0)

ClinicaDL defines a config class associated to this transform:

.. code-block:: python

    >>> from clinicadl.transforms.config import PadConfig
    >>> PadConfig(padding=1)
    PadConfig(padding=1, padding_mode=0.0, name='Pad')

In ClinicaDL, ``PadConfig`` is the equivalent of ``torchio.Pad``.
The user will pass the arguments (e.g. ``padding``) to ``PadConfig``,
and ``PadConfig`` will fetch the default values from
``torchio.Pad`` to complete the configuration (e.g. the default value
of ``padding_mode``).
Then, ClinicaDL objects will take care of converting ``PadConfig`` to
``torchio.Pad`` when they need the transform.

In summary, the user is **encouraged to manipulate configuration classes**,
that are equivalent to the objects they are associated to.

What about objects that don't have a configuration class?
---------------------------------------------------------

However, even if the catalogue of Deep Learning objects that have a
configuration class in ClinicaDL is fairly comprehensive, the user may
want to use objects that are not included.
That's why, to allow full flexibility to the user, ClinicaDL objects
also accept custom objects. For example, let's take ClinicaDL's ``Transforms``,
whose aim is to gather all the transforms applied to the images (see :ref:`transforms`).
``Transforms`` accept configuration classes:

.. code-block:: python

    >>> from clinicadl.transforms import Transforms
    >>> Transforms(image_transforms=[PadConfig(padding=1)])
    Transforms(
        extraction=Image(extract_method="image"),
        image_transforms=[PadConfig(padding=1, padding_mode=0.0, name="Pad")],
        sample_transforms=[],
        augmentations=[],
    )

But the user can also pass directly a transform:

.. code-block:: python

    >>> Transforms(image_transforms=[torchio.Pad(padding=1)])
    Transforms(
        extraction=Image(extract_method="image"),
        image_transforms=[Pad(padding=1, padding_mode=0)],
        sample_transforms=[],
        augmentations=[],
    )

In the two previous examples, the transform applied to images will be
the same. Nevertheless, there is a key difference: when ``PadConfig``
is used, ``Transforms`` will recognize a configuration class and will
be able to save the parameters used, whereas when ``torchio.Pad`` is used,
``Transforms`` cannot access the parameters.

In summary, ClinicaDL is **flexible** and **accepts foreign objects**, but
**using configuration classes enables full reproducibility**.
That's why, before using your own object (transforms, metrics, optimizers, etc.),
you are encouraged to check if there is an object supported in ClinicaDL
that can do the job.

The present API documentation is precisely here to reference all the
Deep Learning objects supported in ClinicaDL, as well as the ClinicaDL
objects that manipulate them.