from pydantic import BaseModel

class ExperimentModel(BaseModel):
    """
    The experiment data model.

    Args:
        BaseModel (_type_): 
            The Base model class.
    """
    name: str
    tracking_uri: str