from paddle.optimizer import Optimizer
from paddle.fluid import core
from paddle.fluid import framework
from paddle.fluid.layer_helper import LayerHelper
import paddle.fluid as fluid

__all__ = ["QHM"]


class QHM(Optimizer):
    r"""
    Simple QHM optimizer with velocity state
    The update equations are as follows:
    .. math::
        & velocity = mu * velocity + gradient
        &\quad   param = param - learning\_rate * ((1-v) * gradient + v * velocity)
    Parameters:
        learning_rate (float|Tensor|LearningRateDecay, optional): The learning rate used to update ``Parameter``.
            It can be a float value, a ``Tensor`` with a float type or a LearningRateDecay. The default value is 0.001.
        momentum (float): Momentum factor. The default value is 0.9.
        parameters (list, optional): List of ``Tensor`` to update to minimize ``loss``. \
            This parameter is required in dygraph mode. \
            The default value is None in static mode, at this time all parameters will be updated.
        weight_decay (float|WeightDecayRegularizer, optional): The strategy of regularization. \
        It canbe a float value as coeff of L2 regularization or \
        :ref:`api_fluid_regularizer_L1Decay`, :ref:`api_fluid_regularizer_L2Decay`.
        If a parameter has set regularizer using :ref:`api_fluid_ParamAttr` already, \
        the regularization setting here in optimizer will be ignored for this parameter. \
        Otherwise, the regularization setting here in optimizer will take effect. \
        Default None, meaning there is no regularization.
        grad_clip (GradientClipBase, optional): Gradient cliping strategy, it's an instance of
            some derived class of ``GradientClipBase`` . There are three cliping strategies
            ( :ref:`api_fluid_clip_GradientClipByGlobalNorm` , :ref:`api_fluid_clip_GradientClipByNorm` ,
            :ref:`api_fluid_clip_GradientClipByValue` ). Default None, meaning there is no gradient clipping.
        name (str, optional): The default value is None. Normally there is no need for user
                to set this property. For more information, please refer to
                :ref:`api_guide_Name` .
    Examples:
        .. code-block:: python
            import paddle
            import numpy as np
            paddle.disable_static()
            inp = np.random.uniform(-0.1, 0.1, [10, 10]).astype("float32")
            linear = paddle.nn.Linear(10, 10)
            inp = paddle.to_tensor(inp)
            out = linear(inp)
            loss = paddle.mean(out)
            beta1 = paddle.to_tensor([0.9], dtype="float32")
            beta2 = paddle.to_tensor([0.99], dtype="float32")
            qhm = QHM(learning_rate=0.1, parameters=linear.parameters(), weight_decay=0.01)
            back = out.backward()
            qhm.step()
            qhm.clear_grad()
    """
    _velocity_acc_str = "velocity"

    def __init__(self,
                 learning_rate=0.001,
                 momentum=0.999,
                 v=0.7,
                 parameters=None,
                 weight_decay=None,
                 grad_clip=None,
                 name=None):
        if learning_rate is None:
            raise ValueError("learning_rate is not set")
        if momentum is None:
            raise ValueError("momentum is not set")
        super(QHM, self).__init__(
            learning_rate=learning_rate,
            parameters=parameters,
            weight_decay=weight_decay,
            grad_clip=grad_clip,
            name=name)
        self.v = v
        self.type = "momentum"
        self._momentum = momentum
        use_nesterov=False
        self._use_nesterov = bool(use_nesterov)
        if framework.in_dygraph_mode():
            self.helper = LayerHelper(self.__class__.__name__)
            for p in parameters:
                self._add_accumulator(self._velocity_acc_str, p)
        else:
            all_parameters = fluid.default_main_program().global_block(
            ).all_parameters()
            self.helper = LayerHelper(self.__class__.__name__)
            for p in all_parameters:
                self._add_accumulator(self._velocity_acc_str, p)

    def _create_accumulators(self, block, parameters):
        assert isinstance(block, framework.Block)
        # create accumulator in init func, so no implementation here

    def _append_optimize_op(self, block, param_and_grad):
        assert isinstance(block, framework.Block)

        velocity_acc = self._get_accumulator(self._velocity_acc_str,
                                             param_and_grad[0])

        lr = self._create_param_lr(param_and_grad)

        if framework.in_dygraph_mode():
            _, _ = core.ops.momentum(param_and_grad[0], param_and_grad[1],
                                     (self.v/self._momentum)*(velocity_acc-param_and_grad[1]), lr, param_and_grad[0],
                                     velocity_acc, 'mu', self._momentum,
                                     'use_nesterov', self._use_nesterov)
            return None

        attrs = {"mu": self._momentum, "use_nesterov": self._use_nesterov}
        inputs = {
            "Param": [param_and_grad[0]],
            "Grad": [param_and_grad[1]],
            "Velocity": [(self.v/self._momentum)*(velocity_acc-param_and_grad[1])],
            "LearningRate": [lr]
        }
     
        outputs = {
            "ParamOut": [param_and_grad[0]],
            "VelocityOut": [velocity_acc]
        }
        # create the qhm optimize op
        momentum_op = block.append_op(
            type=self.type,
            inputs=inputs,
            outputs=outputs,
            attrs=attrs,
            stop_gradient=True)

        return momentum_op