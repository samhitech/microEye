import ast
import threading
import time
import weakref
from typing import Any, Callable

from pyqtgraph.parametertree import Parameter
from pyqtgraph.parametertree.parameterTypes import ActionParameter

from microEye.hardware.cams.camera_panel import Camera_Panel
from microEye.utils.parameter_tree import Tree


class WeakObjects:
    OBJECTS = weakref.WeakSet()

    @classmethod
    def addObject(cls, object):
        cls.OBJECTS.add(object)

    @classmethod
    def removeObject(cls, object):
        cls.OBJECTS.discard(object)

    @classmethod
    def getObject(cls, name: str):
        for ref in cls.OBJECTS:
            if str(ref) == name:
                return ref
        return None

    @classmethod
    def getObjectNames(cls):
        return [str(obj) for obj in cls.OBJECTS]


class BaseAction:
    '''
    Base class for all actions in the editor.

    Attributes
    ----------
    id_counter : int
        Global counter for action IDs.
    NAME : str
        The name of the action type.
    id : int
        The unique identifier for each action instance.
    name : str
        The name of the action instance.
    '''

    id_counter = 1  # Global counter for action IDs
    NAME = 'Base Action'

    def __init__(self):
        '''
        Initialize a new BaseAction instance.
        '''
        self.id = BaseAction.id_counter
        BaseAction.id_counter += 1
        self.name = self.__class__.__name__

    def output(self, text: str, **kwargs):
        output = kwargs.get('output')
        if output is not None:
            output.emit(text, kwargs.get('level', 0))

    def execute(self, **kwargs):
        '''
        Execute the action. To be implemented by subclasses.

        Raises
        ------
        NotImplementedError
            If subclasses do not implement the execute method.
        '''
        raise NotImplementedError('Subclasses must implement the execute method')

    def __str__(self) -> str:
        '''Return a string representation of the action.'''
        return f'{self.name} {self.id}'

    def toHTML(self) -> str:
        '''Return an HTML representation of the action.'''
        return '<span style="color:#0f0;">' + f'<b>{self.name} {self.id}</b></span>'


class FunctionCall(BaseAction):
    NAME = 'Function Call'

    def __init__(self, function: Callable = None, Args: tuple = ()):
        '''
        Initialize a new FunctionCall instance.

        Parameters
        ----------
        function : Callable, optional
            The function to be called, by default None.
        Args : tuple, optional
            The arguments to be passed to the function, by default ().

        Raises
        ------
        TypeError
            If the provided function is not callable.
        '''
        super().__init__()
        self.function = function
        self.function_args = Args

    def execute(self, **kwargs):
        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            self.output(f'{str(self)}: Execution stopped.', **kwargs)
            return

        self.output(f'{str(self)}: Executing...', **kwargs)
        if self.function is not None:
            self.function(*self.function_args, **kwargs)


class ActionGroup(BaseAction):
    '''
    Action that groups multiple child actions.

    Subclass of BaseAction.

    Attributes
    ----------
    NAME : str
        The name of the action type.
    child_actions : list
        List of child actions.
    '''

    NAME = 'Action Group'

    def __init__(self):
        '''
        Initialize a new ActionGroup instance.
        '''
        super().__init__()
        self.child_actions: list[BaseAction] = []

    def execute(self, **kwargs):
        '''
        Execute the action group by executing all child actions.
        '''
        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            self.output(f'{str(self)}: Execution stopped.', **kwargs)
            return

        self.output(f'{str(self)}: Executing...', **kwargs)
        kwargs['level'] = kwargs.get('level', 0) + 1
        for child_action in self.child_actions:
            if event and event.is_set():
                break
            child_action.execute(**kwargs)


class ForLoop(ActionGroup):
    '''
    Action that repeats the execution of its child actions.

    Subclass of ActionGroup.

    Attributes
    ----------
    NAME : str
        The name of the action type.
    repeat_count : int
        The number of times to repeat the child actions.
    '''

    NAME = 'For Loop'

    def __init__(self, repeat_count=3):
        '''
        Initialize a new ForLoop instance.

        Parameters
        ----------
        repeat_count : int, optional
            The number of times to repeat the child actions, by default 3.
        '''
        super().__init__()
        self.repeat_count = repeat_count

    def setRepeatCount(self, repeat_count):
        '''
        Set the number of times to repeat the child actions.

        Parameters
        ----------
        repeat_count : int
            The number of times to repeat the child actions.
        '''
        self.repeat_count = repeat_count

    def execute(self, **kwargs):
        '''
        Execute the for loop by repeating the child actions.

        Notes
        -----
        Overrides the execute method in ActionGroup.
        '''
        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            self.output(f'{str(self)}: Execution stopped.', **kwargs)
            return

        new_kwargs = kwargs.copy()
        new_kwargs['level'] = kwargs.get('level', 0) + 1
        for i in range(self.repeat_count):
            if event and event.is_set():
                break
            self.output(f'{str(self)} ({i + 1})', **kwargs)
            new_kwargs[f'i{self.id}'] = i
            for child_action in self.child_actions:
                if event and event.is_set():
                    break
                child_action.execute(**new_kwargs)

    def __str__(self) -> str:
        '''
        Return a string representation of the for loop.
        '''
        return f'{self.__class__.NAME} {self.id}: range({self.repeat_count})'

    def toHTML(self) -> str:
        '''
        Return a string representation of the for loop.

        Returns
        -------
        str
            String representation of the for loop.
        '''
        return (
            f'<span style="color:#c586c0;">for</span> '
            f'<span style="color:#9cdcfe;">i{self.id}</span> '
            f'<span style="color:#c586c0;">in</span> '
            f'<span style="color:#4ec9b0;">range</span>'
            '<span style="color:#cccccc;">(</span>'
            f'<span style="color:#9cdcfe;">{self.repeat_count}</span>'
            '<span style="color:#cccccc;">)</span>'
            '<span style="color:#cccccc;">:</span>'
        )


class ParameterAdjustmentAction(BaseAction):
    NAME = 'Parameter'

    def __init__(
        self,
        target_object: Tree = None,
        parameter_name: str = 'N/A',
        parameter_value: Any = None,
        delay: float = 0.0,
        is_expression: bool = False,
    ):
        super().__init__()
        self.target_object = target_object
        self.parameter_name = parameter_name
        self.parameter_value = parameter_value
        self.delay = delay
        self.is_expression = is_expression

    def get_paramtree(self):
        target_object = WeakObjects.getObject(self.target_object)

        if target_object is None:
            return None

        if isinstance(target_object, Tree):
            return target_object
        if isinstance(target_object, Camera_Panel):
            return target_object.camera_options
        for attr_name in dir(target_object):
            attr = getattr(target_object, attr_name)
            if isinstance(attr, Tree):
                return attr

    def get_event(self, attr: str):
        target_object = WeakObjects.getObject(self.target_object)

        if hasattr(target_object, attr):
            event = getattr(target_object, attr)
            if isinstance(event, threading.Event):
                return event
            else:
                return None

    def safe_eval(self, expr: str, variables: dict) -> Any:
        # Basic safety check: ensure the expression is a valid Python expression
        try:
            ast.parse(expr, mode='eval')
        except SyntaxError:
            raise

        # Evaluate the expression with only the provided variables
        return eval(expr, {'__builtins__': {}}, variables)

    def execute(self, **kwargs):
        target_object = self.get_paramtree()
        if target_object is None:
            self.output(f'{str(self)}: object not found ...', **kwargs)
            return

        event: threading.Event = kwargs.get('event')
        if event and event.is_set():
            self.output(f'{str(self)}: Execution stopped.', **kwargs)
            return

        self.output(f'{str(self)}: Executing...', **kwargs)
        param = target_object.get_param(self.parameter_name)
        if param is not None:
            if isinstance(param, ActionParameter):
                param.activate()
                if self.parameter_value and 'event' in param.opts:  # Wait for event
                    time.sleep(0.5)
                    param_event = self.get_event(param.opts['event'])
                    if param_event is not None:  # Wait for event
                        param_event.wait()
                        time.sleep(0.2)
            elif isinstance(param, Parameter):
                if self.is_expression:
                    try:
                        value = self.safe_eval(self.parameter_value, kwargs)
                    except Exception as e:
                        self.output(
                            f'{str(self)}: Expression evaluation failed: {str(e)}',
                            **kwargs,
                        )
                        event.set()
                        return
                else:
                    value = self.parameter_value
                param.setValue(value)
        if self.delay > 0:
            self.output(f'{str(self)}: waiting {self.delay:.3f} secs ...', **kwargs)
            # time.sleep(self.delay)
            start_time = time.time()
            while time.time() - start_time < self.delay:
                if event and event.is_set():
                    self.output(
                        f'{str(self)}: Execution stopped during delay.', **kwargs
                    )
                    return
                time.sleep(0.001)  # Short sleep to prevent busy waiting

    def __str__(self) -> str:
        return f'{self.__class__.NAME} {self.id}'

    def toHTML(self) -> str:
        param_name = self.parameter_name.replace('.', ' → ')
        target_object = self.get_paramtree()
        param_type = (
            target_object.get_param(self.parameter_name).type()
            if target_object
            else None
        )
        param_suffix = (
            target_object.get_param(self.parameter_name).opts.get('suffix', '')
            if target_object
            else ''
        )
        if param_type == 'action':
            value_label = '(<span style="color: #0f0">' + (
                'wait</span>)' if self.parameter_value else 'no wait</span>)'
            )
        else:
            value = (
                f'"{self.parameter_value}"'
                if not isinstance(self.parameter_value, (int, float, bool))
                and not self.is_expression
                else f'{self.parameter_value}'
            )
            value_label = (
                '= <span style="color: #0f0">'
                + f"{'expr: ' if self.is_expression else ''}"
                + value
                + f'</span> {param_suffix}'
                if self.parameter_value is not None
                else ''
            )
        return (
            # '<span style="color:#0f0;">'
            # f'<b>{self.__class__.NAME} {self.id}</b></span>:<br>'
            '<span style="color:#9cdcfe;">'
            f'<i>{target_object.__class__.__name__}</i> {self.id}</span>:<br>'
            f'<span style="font-size: 0.8em; color:#4ec9b0;">{param_name}</span> '
            + value_label
            + f'<br><span style="font-size: 0.8em; color:#9cdcfe;">Type</span>'
            f': <span style="color: #0f0">{param_type}</span><br>'
            f'<span style="font-size: 0.8em; color:#9cdcfe;">Delay</span>'
            f': <span style="color: #0f0">{self.delay}</span> Seconds'
        )
