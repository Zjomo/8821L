<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="18008000">
	<Item Name="My Computer" Type="My Computer">
		<Property Name="NI.SortType" Type="Int">3</Property>
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="Command VIs" Type="Folder">
			<Item Name="AbortMotion.vi" Type="VI" URL="../Command VIs/AbortMotion.vi"/>
			<Item Name="AbsoluteMove.vi" Type="VI" URL="../Command VIs/AbsoluteMove.vi"/>
			<Item Name="GetAbsTargetPos.vi" Type="VI" URL="../Command VIs/GetAbsTargetPos.vi"/>
			<Item Name="GetAcceleration.vi" Type="VI" URL="../Command VIs/GetAcceleration.vi"/>
			<Item Name="GetCLDeadband.vi" Type="VI" URL="../Command VIs/GetCLDeadband.vi"/>
			<Item Name="GetCLEnabledSetting.vi" Type="VI" URL="../Command VIs/GetCLEnabledSetting.vi"/>
			<Item Name="GetCLHardwareStatus.vi" Type="VI" URL="../Command VIs/GetCLHardwareStatus.vi"/>
			<Item Name="GetCLThreshold.vi" Type="VI" URL="../Command VIs/GetCLThreshold.vi"/>
			<Item Name="GetCLUnits.vi" Type="VI" URL="../Command VIs/GetCLUnits.vi"/>
			<Item Name="GetCLUpdateInterval.vi" Type="VI" URL="../Command VIs/GetCLUpdateInterval.vi"/>
			<Item Name="GetDeviceAddress.vi" Type="VI" URL="../Command VIs/GetDeviceAddress.vi"/>
			<Item Name="GetErrorMsg.vi" Type="VI" URL="../Command VIs/GetErrorMsg.vi"/>
			<Item Name="GetErrorNum.vi" Type="VI" URL="../Command VIs/GetErrorNum.vi"/>
			<Item Name="GetHostName.vi" Type="VI" URL="../Command VIs/GetHostName.vi"/>
			<Item Name="GetIdentification.vi" Type="VI" URL="../Command VIs/GetIdentification.vi"/>
			<Item Name="GetMotionDone.vi" Type="VI" URL="../Command VIs/GetMotionDone.vi"/>
			<Item Name="GetMotorType.vi" Type="VI" URL="../Command VIs/GetMotorType.vi"/>
			<Item Name="GetPosition.vi" Type="VI" URL="../Command VIs/GetPosition.vi"/>
			<Item Name="GetRelativeSteps.vi" Type="VI" URL="../Command VIs/GetRelativeSteps.vi"/>
			<Item Name="GetVelocity.vi" Type="VI" URL="../Command VIs/GetVelocity.vi"/>
			<Item Name="JogNegative.vi" Type="VI" URL="../Command VIs/JogNegative.vi"/>
			<Item Name="JogPositive.vi" Type="VI" URL="../Command VIs/JogPositive.vi"/>
			<Item Name="MoveToHome.vi" Type="VI" URL="../Command VIs/MoveToHome.vi"/>
			<Item Name="MoveToNextDirIndexNeg.vi" Type="VI" URL="../Command VIs/MoveToNextDirIndexNeg.vi"/>
			<Item Name="MoveToNextDirIndexPos.vi" Type="VI" URL="../Command VIs/MoveToNextDirIndexPos.vi"/>
			<Item Name="MoveToTravelLimitNeg.vi" Type="VI" URL="../Command VIs/MoveToTravelLimitNeg.vi"/>
			<Item Name="MoveToTravelLimitPos.vi" Type="VI" URL="../Command VIs/MoveToTravelLimitPos.vi"/>
			<Item Name="RelativeMove.vi" Type="VI" URL="../Command VIs/RelativeMove.vi"/>
			<Item Name="SaveToMemory.vi" Type="VI" URL="../Command VIs/SaveToMemory.vi"/>
			<Item Name="SetAcceleration.vi" Type="VI" URL="../Command VIs/SetAcceleration.vi"/>
			<Item Name="SetCLDeadband.vi" Type="VI" URL="../Command VIs/SetCLDeadband.vi"/>
			<Item Name="SetCLEnabledSetting.vi" Type="VI" URL="../Command VIs/SetCLEnabledSetting.vi"/>
			<Item Name="SetCLHardwareStatus.vi" Type="VI" URL="../Command VIs/SetCLHardwareStatus.vi"/>
			<Item Name="SetCLThreshold.vi" Type="VI" URL="../Command VIs/SetCLThreshold.vi"/>
			<Item Name="SetCLUnits.vi" Type="VI" URL="../Command VIs/SetCLUnits.vi"/>
			<Item Name="SetCLUpdateInterval.vi" Type="VI" URL="../Command VIs/SetCLUpdateInterval.vi"/>
			<Item Name="SetDeviceAddress.vi" Type="VI" URL="../Command VIs/SetDeviceAddress.vi"/>
			<Item Name="SetHostName.vi" Type="VI" URL="../Command VIs/SetHostName.vi"/>
			<Item Name="SetMotorType.vi" Type="VI" URL="../Command VIs/SetMotorType.vi"/>
			<Item Name="SetVelocity.vi" Type="VI" URL="../Command VIs/SetVelocity.vi"/>
			<Item Name="SetZeroPosition.vi" Type="VI" URL="../Command VIs/SetZeroPosition.vi"/>
			<Item Name="StopMotion.vi" Type="VI" URL="../Command VIs/StopMotion.vi"/>
		</Item>
		<Item Name="Device VIs" Type="Folder">
			<Item Name="DeviceClose.vi" Type="VI" URL="../Device VIs/DeviceClose.vi"/>
			<Item Name="DeviceOpen.vi" Type="VI" URL="../Device VIs/DeviceOpen.vi"/>
			<Item Name="DeviceQuery.vi" Type="VI" URL="../Device VIs/DeviceQuery.vi"/>
			<Item Name="DeviceRead.vi" Type="VI" URL="../Device VIs/DeviceRead.vi"/>
			<Item Name="DeviceWrite.vi" Type="VI" URL="../Device VIs/DeviceWrite.vi"/>
			<Item Name="GetDeviceAddresses.vi" Type="VI" URL="../Device VIs/GetDeviceAddresses.vi"/>
			<Item Name="GetMasterDeviceAddress.vi" Type="VI" URL="../Device VIs/GetMasterDeviceAddress.vi"/>
			<Item Name="GetModelSerial.vi" Type="VI" URL="../Device VIs/GetModelSerial.vi"/>
			<Item Name="InitMultipleDevices.vi" Type="VI" URL="../Device VIs/InitMultipleDevices.vi"/>
			<Item Name="InitSingleDevice.vi" Type="VI" URL="../Device VIs/InitSingleDevice.vi"/>
			<Item Name="IsModel8743.vi" Type="VI" URL="../Device VIs/IsModel8743.vi"/>
			<Item Name="LogFileWrite.vi" Type="VI" URL="../Device VIs/LogFileWrite.vi"/>
			<Item Name="Shutdown.vi" Type="VI" URL="../Device VIs/Shutdown.vi"/>
		</Item>
		<Item Name="Sample8743UI" Type="Folder">
			<Item Name="Sample8743UI.vi" Type="VI" URL="../Sample8743UI/Sample8743UI.vi"/>
			<Item Name="Global Variables.vi" Type="VI" URL="../Sample8743UI/Global Variables.vi"/>
			<Item Name="UIDisable.vi" Type="VI" URL="../Sample8743UI/UIDisable.vi"/>
			<Item Name="GetDiscoveredDevices.vi" Type="VI" URL="../Sample8743UI/GetDiscoveredDevices.vi"/>
			<Item Name="CreateControllerName.vi" Type="VI" URL="../Sample8743UI/CreateControllerName.vi"/>
			<Item Name="UIEnable.vi" Type="VI" URL="../Sample8743UI/UIEnable.vi"/>
			<Item Name="FillControllerCombo.vi" Type="VI" URL="../Sample8743UI/FillControllerCombo.vi"/>
			<Item Name="OnTimeout.vi" Type="VI" URL="../Sample8743UI/OnTimeout.vi"/>
			<Item Name="MotionCheck.vi" Type="VI" URL="../Sample8743UI/MotionCheck.vi"/>
			<Item Name="DisplayPosition.vi" Type="VI" URL="../Sample8743UI/DisplayPosition.vi"/>
			<Item Name="DisplayErrorsForMasterSlave.vi" Type="VI" URL="../Sample8743UI/DisplayErrorsForMasterSlave.vi"/>
			<Item Name="DisplayErrorsForDevice.vi" Type="VI" URL="../Sample8743UI/DisplayErrorsForDevice.vi"/>
			<Item Name="OnDeviceSelected.vi" Type="VI" URL="../Sample8743UI/OnDeviceSelected.vi"/>
			<Item Name="UpdateUIForSelectedDevice.vi" Type="VI" URL="../Sample8743UI/UpdateUIForSelectedDevice.vi"/>
			<Item Name="CloseDevice.vi" Type="VI" URL="../Sample8743UI/CloseDevice.vi"/>
			<Item Name="OnGo.vi" Type="VI" URL="../Sample8743UI/OnGo.vi"/>
			<Item Name="OnStopMotion.vi" Type="VI" URL="../Sample8743UI/OnStopMotion.vi"/>
		</Item>
		<Item Name="SampleGetIDMultiple.vi" Type="VI" URL="../SampleGetIDMultiple.vi"/>
		<Item Name="SampleGetIDSingle.vi" Type="VI" URL="../SampleGetIDSingle.vi"/>
		<Item Name="SampleGetPositionAllSlaves.vi" Type="VI" URL="../SampleGetPositionAllSlaves.vi"/>
		<Item Name="AppendToOutput.vi" Type="VI" URL="../AppendToOutput.vi"/>
		<Item Name="信号光谱11.vi" Type="VI" URL="../信号光谱11.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="NI_ReportGenerationCore.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/NIReport.llb/NI_ReportGenerationCore.lvlib"/>
				<Item Name="NI_report.lvclass" Type="LVClass" URL="/&lt;vilib&gt;/Utility/NIReport.llb/NI_report.lvclass"/>
				<Item Name="Error Cluster From Error Code.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Cluster From Error Code.vi"/>
				<Item Name="Get LV Class Default Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/LVClass/Get LV Class Default Value.vi"/>
				<Item Name="Built App File Layout.vi" Type="VI" URL="/&lt;vilib&gt;/AppBuilder/Built App File Layout.vi"/>
				<Item Name="NI_PackedLibraryUtility.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/LVLibp/NI_PackedLibraryUtility.lvlib"/>
				<Item Name="NI_FileType.lvlib" Type="Library" URL="/&lt;vilib&gt;/Utility/lvfile.llb/NI_FileType.lvlib"/>
				<Item Name="Check if File or Folder Exists.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/libraryn.llb/Check if File or Folder Exists.vi"/>
				<Item Name="NI_Standard Report.lvclass" Type="LVClass" URL="/&lt;vilib&gt;/Utility/NIReport.llb/Standard Report/NI_Standard Report.lvclass"/>
				<Item Name="NI_HTML.lvclass" Type="LVClass" URL="/&lt;vilib&gt;/Utility/NIReport.llb/HTML/NI_HTML.lvclass"/>
				<Item Name="Check File Permissions.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Check File Permissions.vi"/>
				<Item Name="Directory of Top Level VI.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Directory of Top Level VI.vi"/>
				<Item Name="Check Path.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Check Path.vi"/>
				<Item Name="Check Color Table Size.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Check Color Table Size.vi"/>
				<Item Name="Check Data Size.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Check Data Size.vi"/>
				<Item Name="imagedata.ctl" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/imagedata.ctl"/>
				<Item Name="Write PNG File.vi" Type="VI" URL="/&lt;vilib&gt;/picture/png.llb/Write PNG File.vi"/>
				<Item Name="Write JPEG File.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Write JPEG File.vi"/>
				<Item Name="Registry refnum.ctl" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry refnum.ctl"/>
				<Item Name="Registry Handle Master.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry Handle Master.vi"/>
				<Item Name="Close Registry Key.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Close Registry Key.vi"/>
				<Item Name="Create Error Clust.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/Create Error Clust.vi"/>
				<Item Name="Destroy ActiveX Event Queue.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/Destroy ActiveX Event Queue.vi"/>
				<Item Name="OccFireType.ctl" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/OccFireType.ctl"/>
				<Item Name="EventData.ctl" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/EventData.ctl"/>
				<Item Name="Wait On ActiveX Event.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/Wait On ActiveX Event.vi"/>
				<Item Name="Wait types.ctl" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/Wait types.ctl"/>
				<Item Name="Create ActiveX Event Queue.vi" Type="VI" URL="/&lt;vilib&gt;/Platform/ax-events.llb/Create ActiveX Event Queue.vi"/>
				<Item Name="Registry Simplify Data Type.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry Simplify Data Type.vi"/>
				<Item Name="Registry WinErr-LVErr.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry WinErr-LVErr.vi"/>
				<Item Name="Read Registry Value STR.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Read Registry Value STR.vi"/>
				<Item Name="Read Registry Value DWORD.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Read Registry Value DWORD.vi"/>
				<Item Name="Read Registry Value.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Read Registry Value.vi"/>
				<Item Name="Read Registry Value Simple STR.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Read Registry Value Simple STR.vi"/>
				<Item Name="Read Registry Value Simple U32.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Read Registry Value Simple U32.vi"/>
				<Item Name="Read Registry Value Simple.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Read Registry Value Simple.vi"/>
				<Item Name="STR_ASCII-Unicode.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/STR_ASCII-Unicode.vi"/>
				<Item Name="Registry View.ctl" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry View.ctl"/>
				<Item Name="Registry RtKey.ctl" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry RtKey.ctl"/>
				<Item Name="Registry SAM.ctl" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Registry SAM.ctl"/>
				<Item Name="Open Registry Key.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Open Registry Key.vi"/>
				<Item Name="Escape Characters for HTTP.vi" Type="VI" URL="/&lt;vilib&gt;/printing/PathToURL.llb/Escape Characters for HTTP.vi"/>
				<Item Name="Path to URL.vi" Type="VI" URL="/&lt;vilib&gt;/printing/PathToURL.llb/Path to URL.vi"/>
				<Item Name="Search and Replace Pattern.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Search and Replace Pattern.vi"/>
				<Item Name="Generate Temporary File Path.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/libraryn.llb/Generate Temporary File Path.vi"/>
				<Item Name="Flip and Pad for Picture Control.vi" Type="VI" URL="/&lt;vilib&gt;/picture/bmp.llb/Flip and Pad for Picture Control.vi"/>
				<Item Name="Calc Long Word Padded Width.vi" Type="VI" URL="/&lt;vilib&gt;/picture/bmp.llb/Calc Long Word Padded Width.vi"/>
				<Item Name="Write BMP Data To Buffer.vi" Type="VI" URL="/&lt;vilib&gt;/picture/bmp.llb/Write BMP Data To Buffer.vi"/>
				<Item Name="Write BMP Data.vi" Type="VI" URL="/&lt;vilib&gt;/picture/bmp.llb/Write BMP Data.vi"/>
				<Item Name="compatOverwrite.vi" Type="VI" URL="/&lt;vilib&gt;/_oldvers/_oldvers.llb/compatOverwrite.vi"/>
				<Item Name="Write BMP File.vi" Type="VI" URL="/&lt;vilib&gt;/picture/bmp.llb/Write BMP File.vi"/>
				<Item Name="Bit-array To Byte-array.vi" Type="VI" URL="/&lt;vilib&gt;/picture/pictutil.llb/Bit-array To Byte-array.vi"/>
				<Item Name="Create Mask By Alpha.vi" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/Create Mask By Alpha.vi"/>
				<Item Name="Read PNG File.vi" Type="VI" URL="/&lt;vilib&gt;/picture/png.llb/Read PNG File.vi"/>
				<Item Name="NI_Excel.lvclass" Type="LVClass" URL="/&lt;vilib&gt;/Utility/NIReport.llb/Excel/NI_Excel.lvclass"/>
				<Item Name="NI_ReportGenerationToolkit.lvlib" Type="Library" URL="/&lt;vilib&gt;/addons/_office/NI_ReportGenerationToolkit.lvlib"/>
				<Item Name="Read JPEG File.vi" Type="VI" URL="/&lt;vilib&gt;/picture/jpeg.llb/Read JPEG File.vi"/>
				<Item Name="Get File Extension.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/libraryn.llb/Get File Extension.vi"/>
				<Item Name="Handle Open Word or Excel File.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/NIReport.llb/Toolkit/Handle Open Word or Excel File.vi"/>
				<Item Name="Space Constant.vi" Type="VI" URL="/&lt;vilib&gt;/dlg_ctls.llb/Space Constant.vi"/>
				<Item Name="Dynamic To Waveform Array.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/transition.llb/Dynamic To Waveform Array.vi"/>
				<Item Name="Waveform Array To Dynamic.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/transition.llb/Waveform Array To Dynamic.vi"/>
				<Item Name="ex_Modify Signal Name.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/ex_Modify Signal Name.vi"/>
				<Item Name="ex_Modify Signals Names.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/ex_Modify Signals Names.vi"/>
				<Item Name="ex_CorrectErrorChain.vi" Type="VI" URL="/&lt;vilib&gt;/express/express shared/ex_CorrectErrorChain.vi"/>
				<Item Name="NI_MABase.lvlib" Type="Library" URL="/&lt;vilib&gt;/measure/NI_MABase.lvlib"/>
				<Item Name="NI_Matrix.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/Matrix/NI_Matrix.lvlib"/>
				<Item Name="NI_AALBase.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALBase.lvlib"/>
				<Item Name="Find First Error.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find First Error.vi"/>
				<Item Name="NI_Gmath.lvlib" Type="Library" URL="/&lt;vilib&gt;/gmath/NI_Gmath.lvlib"/>
				<Item Name="NI_AALPro.lvlib" Type="Library" URL="/&lt;vilib&gt;/Analysis/NI_AALPro.lvlib"/>
				<Item Name="WDT Number of Waveform Samples DBL.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples DBL.vi"/>
				<Item Name="WDT Number of Waveform Samples SGL.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples SGL.vi"/>
				<Item Name="WDT Number of Waveform Samples I8.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples I8.vi"/>
				<Item Name="WDT Number of Waveform Samples I32.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples I32.vi"/>
				<Item Name="WDT Number of Waveform Samples I16.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples I16.vi"/>
				<Item Name="WDT Number of Waveform Samples EXT.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples EXT.vi"/>
				<Item Name="WDT Number of Waveform Samples CDB.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Number of Waveform Samples CDB.vi"/>
				<Item Name="Number of Waveform Samples.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/Number of Waveform Samples.vi"/>
				<Item Name="WDT Get Waveform Time Array DBL.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/WDT Get Waveform Time Array DBL.vi"/>
				<Item Name="DTbl Digital Size.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/DTblOps.llb/DTbl Digital Size.vi"/>
				<Item Name="DWDT Digital Size.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/DWDTOps.llb/DWDT Digital Size.vi"/>
				<Item Name="DWDT Get Waveform Time Array.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/DWDTOps.llb/DWDT Get Waveform Time Array.vi"/>
				<Item Name="Get Waveform Time Array.vi" Type="VI" URL="/&lt;vilib&gt;/Waveform/WDTOps.llb/Get Waveform Time Array.vi"/>
				<Item Name="subCurveFitting.vi" Type="VI" URL="/&lt;vilib&gt;/express/express analysis/CurveFittingBlock.llb/subCurveFitting.vi"/>
				<Item Name="subBuildXYGraph.vi" Type="VI" URL="/&lt;vilib&gt;/express/express controls/BuildXYGraphBlock.llb/subBuildXYGraph.vi"/>
				<Item Name="GetHelpDir.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetHelpDir.vi"/>
				<Item Name="BuildHelpPath.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/BuildHelpPath.vi"/>
				<Item Name="LVBoundsTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVBoundsTypeDef.ctl"/>
				<Item Name="Get String Text Bounds.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Get String Text Bounds.vi"/>
				<Item Name="Get Text Rect.vi" Type="VI" URL="/&lt;vilib&gt;/picture/picture.llb/Get Text Rect.vi"/>
				<Item Name="Convert property node font to graphics font.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Convert property node font to graphics font.vi"/>
				<Item Name="Longest Line Length in Pixels.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Longest Line Length in Pixels.vi"/>
				<Item Name="LVRectTypeDef.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/miscctls.llb/LVRectTypeDef.ctl"/>
				<Item Name="Three Button Dialog CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog CORE.vi"/>
				<Item Name="Three Button Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Three Button Dialog.vi"/>
				<Item Name="DialogTypeEnum.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogTypeEnum.ctl"/>
				<Item Name="Not Found Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Not Found Dialog.vi"/>
				<Item Name="Set Bold Text.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set Bold Text.vi"/>
				<Item Name="Clear Errors.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Clear Errors.vi"/>
				<Item Name="eventvkey.ctl" Type="VI" URL="/&lt;vilib&gt;/event_ctls.llb/eventvkey.ctl"/>
				<Item Name="TagReturnType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/TagReturnType.ctl"/>
				<Item Name="ErrWarn.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/ErrWarn.ctl"/>
				<Item Name="Details Display Dialog.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Details Display Dialog.vi"/>
				<Item Name="Find Tag.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Find Tag.vi"/>
				<Item Name="Format Message String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Format Message String.vi"/>
				<Item Name="whitespace.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/whitespace.ctl"/>
				<Item Name="Trim Whitespace.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Trim Whitespace.vi"/>
				<Item Name="Error Code Database.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Error Code Database.vi"/>
				<Item Name="GetRTHostConnectedProp.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/GetRTHostConnectedProp.vi"/>
				<Item Name="Set String Value.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Set String Value.vi"/>
				<Item Name="Check Special Tags.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Check Special Tags.vi"/>
				<Item Name="General Error Handler Core CORE.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler Core CORE.vi"/>
				<Item Name="DialogType.ctl" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/DialogType.ctl"/>
				<Item Name="General Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/General Error Handler.vi"/>
				<Item Name="Simple Error Handler.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/error.llb/Simple Error Handler.vi"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="Write Spreadsheet String.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Spreadsheet String.vi"/>
				<Item Name="Write Delimited Spreadsheet (DBL).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Delimited Spreadsheet (DBL).vi"/>
				<Item Name="Write Delimited Spreadsheet (string).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Delimited Spreadsheet (string).vi"/>
				<Item Name="Write Delimited Spreadsheet (I64).vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Delimited Spreadsheet (I64).vi"/>
				<Item Name="Write Delimited Spreadsheet.vi" Type="VI" URL="/&lt;vilib&gt;/Utility/file.llb/Write Delimited Spreadsheet.vi"/>
				<Item Name="NI_PtbyPt.lvlib" Type="Library" URL="/&lt;vilib&gt;/ptbypt/NI_PtbyPt.lvlib"/>
				<Item Name="Query Registry Key Info.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Query Registry Key Info.vi"/>
				<Item Name="Enum Registry Keys.vi" Type="VI" URL="/&lt;vilib&gt;/registry/registry.llb/Enum Registry Keys.vi"/>
			</Item>
			<Item Name="CmdLib.dll" Type="Document" URL="/C/Code/Apps/Utilities/Picomotor/PicomotorApp/Install/Samples/LabVIEW/Model 8743/LabVIEW 2009/CmdLib.dll"/>
			<Item Name="CmdLib.dll" Type="Document" URL="../CmdLib.dll"/>
			<Item Name="lvanlys.dll" Type="Document" URL="/&lt;resource&gt;/lvanlys.dll"/>
			<Item Name="Advapi32.dll" Type="Document" URL="Advapi32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="kernel32.dll" Type="Document" URL="kernel32.dll">
				<Property Name="NI.PreserveRelativePath" Type="Bool">true</Property>
			</Item>
			<Item Name="read detector uniq-id.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/read detector uniq-id.vi"/>
			<Item Name="Initialize CCD.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Initialize CCD.vi"/>
			<Item Name="warning.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/warning.vi"/>
			<Item Name="Init ADC &amp; Gain.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Init ADC &amp; Gain.vi"/>
			<Item Name="Trigger - Get Supported Input.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Trigger - Get Supported Input.vi"/>
			<Item Name="MultiAccu - Is Supported.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/MultiAccu - Is Supported.vi"/>
			<Item Name="Spectral Reformat CCD area.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Spectral Reformat CCD area.vi"/>
			<Item Name="read temperature.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/read temperature.vi"/>
			<Item Name="Setup CCD.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Setup CCD.vi"/>
			<Item Name="Trigger - Disable.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Trigger - Disable.vi"/>
			<Item Name="Trigger - Enable.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Trigger - Enable.vi"/>
			<Item Name="MultiAccu - Disable.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/MultiAccu - Disable.vi"/>
			<Item Name="MultiAccu - Configure.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/MultiAccu - Configure.vi"/>
			<Item Name="MultiAccu - Acquire spectral.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/MultiAccu - Acquire spectral.vi"/>
			<Item Name="Array reset dimension.vi" Type="VI" URL="../instr.lib/USBCCD_Win7_32bit_LV2012/global_lib.llb/Array reset dimension.vi"/>
			<Item Name="Get Monos From Config Browser.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/Get Monos From Config Browser.vi"/>
			<Item Name="JYInitMono.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYInitMono.vi"/>
			<Item Name="Show Current Grating.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/Show Current Grating.vi"/>
			<Item Name="RefnumOrUniqueID.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/RefnumOrUniqueID.vi"/>
			<Item Name="GetDetailsForSelectedTurret.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/GetDetailsForSelectedTurret.vi"/>
			<Item Name="JYGetCurrentWL.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYGetCurrentWL.vi"/>
			<Item Name="JYGetMirrorPos.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYGetMirrorPos.vi"/>
			<Item Name="JYGetSlits.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYGetSlits.vi"/>
			<Item Name="JYTurret.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYTurret.vi"/>
			<Item Name="JYMoveToWavelength.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYMoveToWavelength.vi"/>
			<Item Name="JYMirrorMove.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYMirrorMove.vi"/>
			<Item Name="JYSlits.vi" Type="VI" URL="../instr.lib/USBmonos_Win7_32bit_LV2012/Monochromator Toolkit VI.llb/JYSlits.vi"/>
		</Item>
		<Item Name="Build Specifications" Type="Build"/>
	</Item>
</Project>
