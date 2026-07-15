<?xml version='1.0' encoding='UTF-8'?>
<Project Type="Project" LVVersion="15008000">
	<Property Name="CCSymbols" Type="Str"></Property>
	<Property Name="NI.LV.All.SourceOnly" Type="Bool">false</Property>
	<Property Name="NI.Project.Description" Type="Str">ACC 命令</Property>
	<Item Name="My Computer" Type="My Computer">
		<Property Name="server.app.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.control.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="server.tcp.enabled" Type="Bool">false</Property>
		<Property Name="server.tcp.port" Type="Int">0</Property>
		<Property Name="server.tcp.serviceName" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.tcp.serviceName.default" Type="Str">My Computer/VI Server</Property>
		<Property Name="server.vi.callsEnabled" Type="Bool">true</Property>
		<Property Name="server.vi.propertiesEnabled" Type="Bool">true</Property>
		<Property Name="specify.custom.address" Type="Bool">false</Property>
		<Item Name="SubVIs" Type="Folder" URL="/&lt;instrlib&gt;/MMC-100/SubVIs">
			<Property Name="NI.DISK" Type="Bool">true</Property>
		</Item>
		<Item Name="A Simple Control Panel Example.vi" Type="VI" URL="/&lt;instrlib&gt;/MMC-100/A Simple Control Panel Example.vi"/>
		<Item Name="Dependencies" Type="Dependencies">
			<Item Name="vi.lib" Type="Folder">
				<Item Name="VISA Configure Serial Port" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port"/>
				<Item Name="VISA Configure Serial Port (Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Instr).vi"/>
				<Item Name="VISA Configure Serial Port (Serial Instr).vi" Type="VI" URL="/&lt;vilib&gt;/Instr/_visa.llb/VISA Configure Serial Port (Serial Instr).vi"/>
			</Item>
		</Item>
		<Item Name="Build Specifications" Type="Build">
			<Item Name="MMC-100 Example" Type="EXE">
				<Property Name="App_copyErrors" Type="Bool">true</Property>
				<Property Name="App_INI_aliasGUID" Type="Str">{EF29A6BD-0861-4924-9C5C-C86E538713AB}</Property>
				<Property Name="App_INI_GUID" Type="Str">{22CC819F-133E-4AE0-B06E-DFAA8BBA207E}</Property>
				<Property Name="App_serverConfig.httpPort" Type="Int">8002</Property>
				<Property Name="Bld_autoIncrement" Type="Bool">true</Property>
				<Property Name="Bld_buildCacheID" Type="Str">{A3A9F95F-BF76-4698-B214-C75AC16E553D}</Property>
				<Property Name="Bld_buildSpecName" Type="Str">MMC-100 Example</Property>
				<Property Name="Bld_excludeLibraryItems" Type="Bool">true</Property>
				<Property Name="Bld_excludePolymorphicVIs" Type="Bool">true</Property>
				<Property Name="Bld_localDestDir" Type="Path">../builds/NI_AB_PROJECTNAME/MMC-100 Example</Property>
				<Property Name="Bld_localDestDirType" Type="Str">relativeToCommon</Property>
				<Property Name="Bld_modifyLibraryFile" Type="Bool">true</Property>
				<Property Name="Bld_previewCacheID" Type="Str">{72DB5D86-9296-40BD-9C3C-9C67E6F5DC6E}</Property>
				<Property Name="Bld_version.build" Type="Int">4</Property>
				<Property Name="Bld_version.major" Type="Int">1</Property>
				<Property Name="Destination[0].destName" Type="Str">MMC100.exe</Property>
				<Property Name="Destination[0].path" Type="Path">../builds/NI_AB_PROJECTNAME/MMC-100 Example/MMC100.exe</Property>
				<Property Name="Destination[0].preserveHierarchy" Type="Bool">true</Property>
				<Property Name="Destination[0].type" Type="Str">App</Property>
				<Property Name="Destination[1].destName" Type="Str">Support Directory</Property>
				<Property Name="Destination[1].path" Type="Path">../builds/NI_AB_PROJECTNAME/MMC-100 Example</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="Source[0].itemID" Type="Str">{2A614DF0-E48F-4619-89E5-7526A642558A}</Property>
				<Property Name="Source[0].type" Type="Str">Container</Property>
				<Property Name="Source[1].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[1].itemID" Type="Ref">/My Computer/A Simple Control Panel Example.vi</Property>
				<Property Name="Source[1].sourceInclusion" Type="Str">TopLevel</Property>
				<Property Name="Source[1].type" Type="Str">VI</Property>
				<Property Name="Source[2].destinationIndex" Type="Int">0</Property>
				<Property Name="Source[2].itemID" Type="Ref">/My Computer/SubVIs/COM.txt</Property>
				<Property Name="Source[2].sourceInclusion" Type="Str">Include</Property>
				<Property Name="SourceCount" Type="Int">3</Property>
				<Property Name="TgtF_companyName" Type="Str">Micronix</Property>
				<Property Name="TgtF_fileDescription" Type="Str">MMC-100 Example</Property>
				<Property Name="TgtF_internalName" Type="Str">MMC-100 Example</Property>
				<Property Name="TgtF_legalCopyright" Type="Str">Copyright ? 2015 Micronix</Property>
				<Property Name="TgtF_productName" Type="Str">MMC-100 Example</Property>
				<Property Name="TgtF_targetfileGUID" Type="Str">{94875344-152E-4F0A-9F83-86E843A17C4C}</Property>
				<Property Name="TgtF_targetfileName" Type="Str">MMC100.exe</Property>
			</Item>
			<Item Name="MMC-100 Example Installer" Type="Installer">
				<Property Name="Destination[0].name" Type="Str">Micronix</Property>
				<Property Name="Destination[0].parent" Type="Str">{3912416A-D2E5-411B-AFEE-B63654D690C0}</Property>
				<Property Name="Destination[0].tag" Type="Str">{1B237511-05CF-4539-BC9F-A4865F8C29B8}</Property>
				<Property Name="Destination[0].type" Type="Str">userFolder</Property>
				<Property Name="Destination[1].name" Type="Str">MMC-100 example</Property>
				<Property Name="Destination[1].parent" Type="Str">{1B237511-05CF-4539-BC9F-A4865F8C29B8}</Property>
				<Property Name="Destination[1].tag" Type="Str">{5F7F7A1C-7337-4DF6-AD60-296EBBC2DC89}</Property>
				<Property Name="Destination[1].type" Type="Str">userFolder</Property>
				<Property Name="DestinationCount" Type="Int">2</Property>
				<Property Name="DistPart[0].flavorID" Type="Str">_deployment_</Property>
				<Property Name="DistPart[0].productID" Type="Str">{C13DF63E-9B02-4CA4-B4CA-3E9B56EFB217}</Property>
				<Property Name="DistPart[0].productName" Type="Str">NI-VISA Runtime 14.0</Property>
				<Property Name="DistPart[0].upgradeCode" Type="Str">{8627993A-3F66-483C-A562-0D3BA3F267B1}</Property>
				<Property Name="DistPart[1].flavorID" Type="Str">DefaultFull</Property>
				<Property Name="DistPart[1].productID" Type="Str">{A8864B2C-7242-4C11-9D65-E88D200A66DE}</Property>
				<Property Name="DistPart[1].productName" Type="Str">NI LabVIEW运行引擎2015 SP1 f7</Property>
				<Property Name="DistPart[1].upgradeCode" Type="Str">{CA8FF739-2EDA-4134-9A70-0F5DD933FDED}</Property>
				<Property Name="DistPartCount" Type="Int">2</Property>
				<Property Name="INST_author" Type="Str">CalAmp</Property>
				<Property Name="INST_autoIncrement" Type="Bool">true</Property>
				<Property Name="INST_buildLocation" Type="Path">../builds/MMC-100 example/MMC-100 Example Installer</Property>
				<Property Name="INST_buildLocation.type" Type="Str">relativeToCommon</Property>
				<Property Name="INST_buildSpecName" Type="Str">MMC-100 Example Installer</Property>
				<Property Name="INST_defaultDir" Type="Str">{5F7F7A1C-7337-4DF6-AD60-296EBBC2DC89}</Property>
				<Property Name="INST_productName" Type="Str">MMC-100 example</Property>
				<Property Name="INST_productVersion" Type="Str">1.0.4</Property>
				<Property Name="InstSpecBitness" Type="Str">32-bit</Property>
				<Property Name="InstSpecVersion" Type="Str">15018017</Property>
				<Property Name="MSI_arpCompany" Type="Str">Micronix</Property>
				<Property Name="MSI_arpContact" Type="Str">info@micronixusa.com</Property>
				<Property Name="MSI_arpPhone" Type="Str">(949) 480-0538</Property>
				<Property Name="MSI_arpURL" Type="Str">http://www.micronixusa.com/</Property>
				<Property Name="MSI_distID" Type="Str">{0F8CD4F5-CE85-4C0E-B737-47ECFDE6D8A3}</Property>
				<Property Name="MSI_osCheck" Type="Int">2</Property>
				<Property Name="MSI_upgradeCode" Type="Str">{85E7AE48-74A5-4FA1-850E-DE29A96AD22C}</Property>
				<Property Name="RegDest[0].dirName" Type="Str">Software</Property>
				<Property Name="RegDest[0].dirTag" Type="Str">{DDFAFC8B-E728-4AC8-96DE-B920EBB97A86}</Property>
				<Property Name="RegDest[0].parentTag" Type="Str">2</Property>
				<Property Name="RegDestCount" Type="Int">1</Property>
				<Property Name="Source[0].dest" Type="Str">{5F7F7A1C-7337-4DF6-AD60-296EBBC2DC89}</Property>
				<Property Name="Source[0].File[0].dest" Type="Str">{5F7F7A1C-7337-4DF6-AD60-296EBBC2DC89}</Property>
				<Property Name="Source[0].File[0].name" Type="Str">MMC100.exe</Property>
				<Property Name="Source[0].File[0].Shortcut[0].destIndex" Type="Int">1</Property>
				<Property Name="Source[0].File[0].Shortcut[0].name" Type="Str">MMC-100 Example</Property>
				<Property Name="Source[0].File[0].Shortcut[0].subDir" Type="Str"></Property>
				<Property Name="Source[0].File[0].Shortcut[1].destIndex" Type="Int">0</Property>
				<Property Name="Source[0].File[0].Shortcut[1].name" Type="Str">MMC-100 Example</Property>
				<Property Name="Source[0].File[0].Shortcut[1].subDir" Type="Str">Micronix</Property>
				<Property Name="Source[0].File[0].ShortcutCount" Type="Int">2</Property>
				<Property Name="Source[0].File[0].tag" Type="Str">{94875344-152E-4F0A-9F83-86E843A17C4C}</Property>
				<Property Name="Source[0].FileCount" Type="Int">1</Property>
				<Property Name="Source[0].name" Type="Str">MMC-100 Example</Property>
				<Property Name="Source[0].tag" Type="Ref">/My Computer/Build Specifications/MMC-100 Example</Property>
				<Property Name="Source[0].type" Type="Str">EXE</Property>
				<Property Name="SourceCount" Type="Int">1</Property>
			</Item>
		</Item>
	</Item>
</Project>
