"""PICam SDK 常量与枚举。

这些常量基于 PICam Programmer's Manual 与公开资料整理。
完整参数列表请参考 PICam 官方文档。
"""

# ---------------------------------------------------------------------------
# PicamError 枚举
# ---------------------------------------------------------------------------
class PicamError:
    None_ = 0
    UnexpectedError = 1
    InvalidPointer = 2
    InvalidCount = 3
    InvalidParameteri = 4
    InvalidHandle = 5
    DynamicRange = 6
    EnumInvalid = 7
    EnumNotSupported = 8
    ValueInvalid = 9
    ValueNotSupported = 10
    ValueNotAllowed = 11
    ValueOutOfRange = 12
    ParameterNotOnline = 13
    ParameterNotReadable = 14
    ParameterNotWritable = 15
    ParameterDoesNotExist = 16
    ParameterHasNotBeenInitialized = 17
    ParameterAlreadyInitialized = 18
    ParameterCurrentlyLocked = 19
    ParameterCouldNotBeCommunicated = 20
    InvalidParameterID = 21
    InvalidEnumeratedType = 22
    InvalidEnumeratorValue = 23
    InvalidDataFormat = 24
    InvalidROI = 25
    DataNotAvailable = 26
    DataNotAvailableTimeout = 27
    DataNotAvailableIncompatibleModel = 28
    DataNotAvailableParameterNotOnline = 29
    DataNotAvailableInvalidParameterValue = 30
    HardwareNotAvailable = 31
    AcquisitionInProgress = 32
    AcquisitionNotInProgress = 33
    Timeout = 34
    AcquisitionUpdatedTimeout = 35
    InvalidParameterValues = 36


ERROR_MESSAGES = {
    PicamError.None_: "Success",
    PicamError.UnexpectedError: "Unexpected error",
    PicamError.InvalidPointer: "Invalid pointer",
    PicamError.InvalidCount: "Invalid count",
    PicamError.InvalidParameteri: "Invalid parameter index",
    PicamError.InvalidHandle: "Invalid handle",
    PicamError.DynamicRange: "Value out of dynamic range",
    PicamError.EnumInvalid: "Invalid enum value",
    PicamError.EnumNotSupported: "Enum value not supported",
    PicamError.ValueInvalid: "Invalid value",
    PicamError.ValueNotSupported: "Value not supported",
    PicamError.ValueNotAllowed: "Value not allowed",
    PicamError.ValueOutOfRange: "Value out of range",
    PicamError.ParameterNotOnline: "Parameter not online",
    PicamError.ParameterNotReadable: "Parameter not readable",
    PicamError.ParameterNotWritable: "Parameter not writable",
    PicamError.ParameterDoesNotExist: "Parameter does not exist",
    PicamError.ParameterHasNotBeenInitialized: "Parameter has not been initialized",
    PicamError.ParameterAlreadyInitialized: "Parameter already initialized",
    PicamError.ParameterCurrentlyLocked: "Parameter currently locked",
    PicamError.ParameterCouldNotBeCommunicated: "Parameter could not be communicated",
    PicamError.InvalidParameterID: "Invalid parameter ID",
    PicamError.InvalidEnumeratedType: "Invalid enumerated type",
    PicamError.InvalidEnumeratorValue: "Invalid enumerator value",
    PicamError.InvalidDataFormat: "Invalid data format",
    PicamError.InvalidROI: "Invalid ROI",
    PicamError.DataNotAvailable: "Data not available",
    PicamError.DataNotAvailableTimeout: "Data not available due to timeout",
    PicamError.DataNotAvailableIncompatibleModel: "Data not available: incompatible model",
    PicamError.DataNotAvailableParameterNotOnline: "Data not available: parameter not online",
    PicamError.DataNotAvailableInvalidParameterValue: "Data not available: invalid parameter value",
    PicamError.HardwareNotAvailable: "Hardware not available",
    PicamError.AcquisitionInProgress: "Acquisition already in progress",
    PicamError.AcquisitionNotInProgress: "Acquisition not in progress",
    PicamError.Timeout: "Timeout",
    PicamError.AcquisitionUpdatedTimeout: "Acquisition update timeout",
    PicamError.InvalidParameterValues: "Invalid parameter values",
}


# ---------------------------------------------------------------------------
# PicamValueType 枚举
# ---------------------------------------------------------------------------
class PicamValueType:
    Integer = 1
    Boolean = 2
    Enumeration = 3
    LargeInteger = 4
    FloatingPoint = 5
    Rois = 6
    Pulse = 7
    Modulations = 8


# ---------------------------------------------------------------------------
# 常用 PicamParameter 参数 ID
# ---------------------------------------------------------------------------
class PicamParameter:
    # 采集
    ExposureTime = 0x01000001
    ShutterTimingMode = 0x01000002
    ShutterClosingDelay = 0x01000003
    IntensifierStatus = 0x01000004
    IntensifierGain = 0x01000005
    GateTracking = 0x01000006
    GateTrackingDelay = 0x01000007
    GateTrackingWidth = 0x01000008
    GatingMode = 0x01000009
    GateWidth = 0x0100000A
    GateDelay = 0x0100000B

    # 传感器
    SensorTemperatureSetPoint = 0x02000001
    SensorTemperatureReading = 0x02000002
    SensorTemperatureStatus = 0x02000003
    ReadoutControlMode = 0x02000004
    ReadoutPortCount = 0x02000005
    ReadoutRateCalculation = 0x02000006
    ReadoutTimeCalculation = 0x02000007

    # ROI
    ActiveWidth = 0x03000001
    ActiveHeight = 0x03000002
    ActiveLeftMargin = 0x03000003
    ActiveRightMargin = 0x03000004
    ActiveTopMargin = 0x03000005
    ActiveBottomMargin = 0x03000006
    ActiveExtendedHeight = 0x03000007
    ActiveExtendedWidth = 0x03000008
    VerticalShiftRate = 0x03000009

    # ADC
    AdcSpeed = 0x04000001
    AdcAnalogGain = 0x04000002
    AdcEMGain = 0x04000003
    AdcQuality = 0x04000004
    CorrectPixelBias = 0x04000005

    # 硬件 I/O
    TriggerSource = 0x05000001
    TriggerResponse = 0x05000002
    TriggerDetermination = 0x05000003
    OutputSignal = 0x05000004
    InvertOutputSignal = 0x05000005


# ---------------------------------------------------------------------------
# 常用枚举值
# ---------------------------------------------------------------------------
class PicamReadoutControlMode:
    FullFrame = 1
    FrameTransfer = 2
    Kinetics = 3
    Custom = 4


class PicamAdcAnalogGain:
    Low = 1
    Medium = 2
    High = 3


class PicamAdcQuality:
    LowNoise = 1
    HighCapacity = 2
    HighSpeed = 3


class PicamTriggerResponse:
    NoResponse = 1
    ReadoutPerTrigger = 2
    StartOnSingleTrigger = 3
    ReadoutPerExternalTrigger = 4


class PicamTriggerDetermination:
    PositivePolarity = 1
    NegativePolarity = 2
    RisingEdge = 3
    FallingEdge = 4


class PicamOutputSignal:
    NotReady = 1
    ShutterOpen = 2
    Busy = 3
    ReadingOut = 4
    Acquiring = 5


class PicamGatingMode:
    Disabled = 1
    External = 2
    Internal = 3


class PicamTemperatureStatus:
    Unlocked = 1
    Locked = 2
    Faulted = 3


class PicamShutterTimingMode:
    Normal = 1
    AlwaysClosed = 2
    AlwaysOpen = 3
    OpenBeforeTrigger = 4
