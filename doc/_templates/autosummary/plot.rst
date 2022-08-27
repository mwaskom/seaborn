{{ fullname | escape | underline}}

.. currentmodule:: {{ module }}

.. autoclass:: {{ objname }}

   {% block methods %}

   .. rubric:: Specification methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      add
      scale

   .. rubric:: Subplot methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      facet
      pair

   .. rubric:: Customization methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      layout
      label
      limit
      share
      theme

   .. rubric:: Integration methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      on

   .. rubric:: Output methods

   .. autosummary::
      :toctree: ./
      :nosignatures:

      plot
      save
      show

   {% endblock %}
