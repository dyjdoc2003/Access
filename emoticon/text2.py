# -*- coding: utf-8 -*-
"""
Created on Wed Nov 11 17:39:24 2020

@author: Tai
"""

#!/usr/bin/python
# -*- coding: utf-8 -*- 

import importlib
import datetime
import csv
import os
import sys

try: 
    import xml.etree.cElementTree as ET 
except ImportError: 
    import xml.etree.ElementTree as ET 

def printf(strings):
    now = datetime.datetime.now()
    now_string=now.strftime('%Y-%m-%d_%H:%M:%S')
    print ("["+now_string+"] " +strings)
    return "["+now_string+"] " +strings


def getCSV_BS(inputfile, outputfile):
    #CSV表头
    FIELDS = ['ServiceArea','Version', 'IPVersion', 'MLDVersion', 'NumOfTS',
              'TSId','MediaPortNumber','GroupAddress','SourceAddress','CaBroadcasterGroupId','LinkageDescriptorURL','NumOfFEC','FECMode','FECModeInfo','MaximumTSBitRate',
              'NumOfService','ServiceName','ServiceId','LicenseId','TierBitMask','ServiceType','StreamType','KeyPuID','ContractCrid','UncontractCrid','RemoteControlKeyId',
              'RenderingObligation','RenderingObligationRa','SIMulticast_MediaPortNumber','SIMulticast_GroupAddress','SIMulticast_SourceAddress']
    f1 = open(outputfile,"wb")
    writer = csv.DictWriter(f1, fieldnames=FIELDS)
    writer.writerow(dict(zip(FIELDS, FIELDS))) 
    f1.close()      

    try: 
        tree = ET.parse(inputfile)     #打开xml文档 
        root = tree.getroot()         #获得root节点  
    except Exception as e: 
        print ("Error:cannot parse file:cfg.xml.")
        print (e)
        sys.exit(1) 

    ServiceArea = root.find('ServiceArea').text
    Version = root.find('Version').text
    IPVersion = root.find('IPVersion').text
    MLDVersion = root.find('MLDVersion').text
    NumOfTS = root.find('NumOfTS').text

    SIMulticast_MediaPortNumber = root.find('SIMulticast').find('MediaPortNumber').text
    SIMulticast_GroupAddress = root.find('SIMulticast').find('GroupAddress').text
    SIMulticast_SourceAddress = root.find('SIMulticast').find('SourceAddress').text

    children =root.findall('TSList')[0]
    for period in children.findall('TS'): #找到root节点下的所有period节点 
        row = {}

        if period.find('TSId') != None:
            TSId = period.find('TSId').text
            TSId = int(TSId,16)

        if period.find('MediaPortNumber') != None:
            MediaPortNumber = period.find('MediaPortNumber').text
        else:
            MediaPortNumber = ''

        if period.find('GroupAddress') != None:
            GroupAddress = period.find('GroupAddress').text
        else:
            GroupAddress = ''

        if period.find('SourceAddress') != None:
            SourceAddress = period.find('SourceAddress').text
        else:
            SourceAddress = ''

        if period.find('CaBroadcasterGroupId') != None:
            CaBroadcasterGroupId = period.find('CaBroadcasterGroupId').text
        else:
            CaBroadcasterGroupId = ''

        if period.find('LinkageDescriptorURL') != None:
            LinkageDescriptorURL = period.find('LinkageDescriptorURL').text
        else:
            LinkageDescriptorURL = ''

        if period.find('NumOfFEC') != None:
            NumOfFEC = period.find('NumOfFEC').text
        else:
            NumOfFEC = ''

        if period.find('FECList') != None :
            FEC = period.find('FECList').find('FEC')
            FECMode = FEC.find('FECMode').text
            FECModeInfo = FEC.find('FECModeInfo').text
        else:
            FECMode = ''
            FECModeInfo = ''

        MaximumTSBitRate = period.find('MaximumTSBitRate').text
        NumOfService = period.find('NumOfService').text

        Service = period.find('ServiceList').find('Service')
        if Service.find('ServiceName') != None:
            ServiceName = Service.find('ServiceName').text
        else:
            ServiceName = ''

        if Service.find('ServiceId') != None:
            ServiceId = Service.find('ServiceId').text
        else:
            ServiceId = ''

        if Service.find('LicenseId') != None:
            LicenseId = Service.find('LicenseId').text
        else:
            LicenseId = ''

        if Service.find('TierBitMask') != None:
            TierBitMask = Service.find('TierBitMask').text
        else:
            TierBitMask = ''

        if Service.find('ServiceType') != None:
            ServiceType = Service.find('ServiceType').text
        else:
            ServiceType = ''

        if Service.find('KeyPuID') != None:
            KeyPuID = Service.find('KeyPuID').text
        else:
            KeyPuID = ''

        if Service.find('ContractCrid') != None:
            ContractCrid = Service.find('ContractCrid').text
        else:
            ContractCrid = ''

        if Service.find('UncontractCrid') != None:
            UncontractCrid = Service.find('UncontractCrid').text
        else:
            UncontractCrid = ''

        if Service.find('RemoteControlKeyId') != None:
            RemoteControlKeyId = Service.find('RemoteControlKeyId').text
        else:
            RemoteControlKeyId = ''

        if Service.find('StreamType') != None:
            StreamType = Service.find('StreamType').text
        else:
            StreamType = ''

        if Service.find('RenderingObligation') != None:
            RenderingObligation = Service.find('RenderingObligation').text
        else:
            RenderingObligation = ''

        if Service.find('RenderingObligationRa') != None:
            RenderingObligationRa = Service.find('RenderingObligationRa').text
        else:
            RenderingObligationRa = ''

        row["ServiceArea"] = ServiceArea
        row["Version"] = Version
        row["IPVersion"] = IPVersion
        row["MLDVersion"] = MLDVersion
        row["NumOfTS"] = NumOfTS

        row["TSId"] = TSId
        row["MediaPortNumber"] = MediaPortNumber
        row["GroupAddress"] = GroupAddress
        row["SourceAddress"] = SourceAddress
        row["CaBroadcasterGroupId"] = CaBroadcasterGroupId
        row["LinkageDescriptorURL"] = LinkageDescriptorURL
        row["NumOfFEC"] = NumOfFEC
        row["FECMode"] = FECMode
        row["FECModeInfo"] = FECModeInfo
        row["MaximumTSBitRate"] = MaximumTSBitRate
        row["NumOfService"] = NumOfService
        row["ServiceName"] = ServiceName
        row["ServiceId"] = ServiceId
        row["LicenseId"] = LicenseId
        row["TierBitMask"] = TierBitMask
        row["ServiceType"] = ServiceType
        row["KeyPuID"] = KeyPuID
        row["ContractCrid"] = ContractCrid
        row["UncontractCrid"] = UncontractCrid
        row["RemoteControlKeyId"] = RemoteControlKeyId
        row["StreamType"] = StreamType
        row["RenderingObligation"] = RenderingObligation
        row["RenderingObligationRa"] = RenderingObligationRa

        row["SIMulticast_MediaPortNumber"] = SIMulticast_MediaPortNumber
        row["SIMulticast_GroupAddress"] = SIMulticast_GroupAddress
        row["SIMulticast_SourceAddress"] = SIMulticast_SourceAddress

        #生成csv
        f1 = open(outputfile, 'ab')
        dict_writer = csv.DictWriter(f1, fieldnames=FIELDS)
        dict_writer.writerow(row)
        f1.close()    


if __name__ == '__main__': 
    importlib.reload(sys)
    file_bs = "test3.xml"
    outputfile_bs = "test.csv" 
    try:
        getCSV_BS(file_bs, outputfile_bs)
        printf("test.xml success")
    except Exception as e:
        print (e)
        printf("test.xml fail")

    os.system("pause")