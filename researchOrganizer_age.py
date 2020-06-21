import pickle
from random import shuffle

import numpy
from keras.engine.saving import load_model
from keras_preprocessing.sequence import pad_sequences

import lstm.classification.lstm_simple
import lstm.classification.lstm_dropout_1
import lstm.classification.lstm_dropout_2
import lstm.classification.lstm_with_cnn
import blstm.classification.blstm_simple
import blstm.classification.blstm_dropout_1
import blstm.classification.blstm_dropout_2
import blstm.classification.blstm_with_cnn


def transform_age_category_5(corpus, topic):
    for post in corpus:
        age = int(post["age"])
        forums_thematic = post["forums_thematic"]
        if 19 < age < 30 and forums_thematic == topic:
            post["age"] = [1, 0, 0, 0, 0]
        elif 29 < age < 40 and forums_thematic == topic:
            post["age"] = [0, 1, 0, 0, 0]
        elif 39 < age < 50 and forums_thematic == topic:
            post["age"] = [0, 0, 1, 0, 0]
        elif 49 < age < 60 and forums_thematic == topic:
            post["age"] = [0, 0, 0, 1, 0]
        elif 59 < age < 70 and forums_thematic == topic:
            post["age"] = [0, 0, 0, 0, 1]
        else:
            post["age"] = "to_del"
    return corpus


def transform_age_category_2(corpus, topic):
    for post in corpus:
        age = int(post["age"])
        forums_thematic = post["forums_thematic"]
        if age < 30 and forums_thematic == topic:
            post["age"] = numpy.int64(0)
        elif 49 < age < 70 and forums_thematic == topic:
            post["age"] = numpy.int64(1)
        else:
            post["age"] = "to_del"
    return corpus


def create_x_y(corpus, task):
    x = []
    y = []
    for message in corpus:
        x.append(message["text_sequence"])
        y.append(message[task])
    return numpy.array(x), numpy.array(y)


def resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name, actual_epochs):
    # load the saved model
    saved_model = load_model(save_model_name)
    _, train_acc = saved_model.evaluate(X_train, y_train, verbose=0)
    _, valid_acc = saved_model.evaluate(X_validation, y_validation, verbose=0)
    result_str = '\t\tAttempt ' + str(save_model_name[-1]) + ", Epochs " + str(actual_epochs) + "\t\t\t"
    result_str += 'Train: %.3f%%, Validation: %.3f%%' % (train_acc*100, valid_acc*100)
    result_str += '\n'
    return result_str


def resultsOfTest(X_test, y_test, save_model_name):
    # load the saved model
    saved_model = load_model(save_model_name)
    _, test_acc = saved_model.evaluate(X_test, y_test, verbose=0)
    result_str = '\t\tAttempt ' + str(save_model_name[-1]) + "\t\t\t"
    result_str += 'Test: %.3f%%' % (test_acc*100)
    result_str += '\n'
    return result_str

# main variables notation
max_review_length = 100
word_embedding_dict = "twitter"
task = "age"
topic = "Watches"
numNeurons = [5, 10, 25, 48, 50, 100, 129]
batch_size = 500
early_stopping_wait = 100
repeat = 1
num_categories = 5
filePathMainInfoTrain = "results/results_age_in-domain_" + str(topic) + "_" + str(word_embedding_dict) + "_max_length_" + str(max_review_length) + ".txt"

"""
training_user_id = ['804', '1783', '330', '68', '1268', '1094', '1556', '296', '384', '1649', '1286', '1924', '734',
                    '1175', '476', '1390', '300', '1327', '1567', '395', '1980', '1259', '1407', '1796', '494',
                    '169', '869', '284', '1570', '1234', '593', '1949', '447', '1926', '986', '1794', '1254',
                    '1134', '1977', '1113', '175', '1604', '638', '1614', '1033', '306', '317', '1212', '978',
                    '1834', '1885', '52', '1079', '492', '651', '150', '1878', '667', '771', '1037', '28', '675',
                    '837', '1749', '1502', '1526', '154', '1048', '1473', '954', '657', '1347', '167', '259',
                    '1322', '1477', '1809', '671', '1889', '1089', '136', '1264', '90', '876', '1806', '1755',
                    '987', '1164', '1678', '1043', '890', '1841', '1955', '220', '430', '206', '993', '1198', '410',
                    '904', '106', '257', '795', '1653', '1191', '1672', '1042', '1215', '857', '235', '1131', '976',
                    '682', '482', '624', '895', '190', '1748', '1543', '485', '1652', '59', '1008', '810', '1818',
                    '950', '905', '458', '1035', '569', '572', '1874', '1591', '755', '1091', '337', '1132', '419',
                    '484', '1702', '841', '668', '1788', '196', '107', '1659', '1491', '533', '1450', '477', '1837',
                    '1557', '1635', '1840', '1324', '171', '188', '1722', '1314', '294', '1380', '1879', '994',
                    '626', '101', '992', '729', '99', '1478', '909', '645', '827', '1573', '952', '829', '316',
                    '250', '1087', '1548', '1369', '1510', '467', '62', '1325', '1163', '1707', '820', '1399',
                    '1817', '1518', '1830', '920', '299', '923', '562', '1929', '1223', '242', '1991', '860',
                    '1691', '563', '478', '1660', '376', '1495', '97', '635', '605', '354', '1019', '1197', '15',
                    '1737', '1729', '1252', '962', '1801', '1415', '1505', '1625', '1374', '641', '100', '1153',
                    '142', '437', '706', '1417', '1423', '874', '1281', '1353', '1723', '708', '279', '1412',
                    '1869', '1867', '1566', '1216', '507', '310', '366', '948', '199', '1975', '1372', '884',
                    '1602', '1995', '1045', '418', '1270', '1844', '558', '1638', '1699', '1881', '1541', '941',
                    '750', '808', '207', '516', '754', '712', '864', '1654', '1835', '692', '1757', '845', '1341',
                    '509', '1220', '1515', '49', '1406', '1684', '1155', '1179', '889', '1599', '1342', '885',
                    '1157', '251', '265', '1745', '1950', '1483', '1121', '292', '240', '39', '1206', '1051', '934',
                    '1807', '988', '231', '1479', '163', '1226', '887', '1994', '409', '119', '1776', '1282', '127',
                    '1364', '1657', '1050', '926', '1762', '139', '842', '1873', '1891', '1468', '1269', '1265',
                    '445', '1802', '1320', '1524', '1114', '580', '96', '1890', '63', '1733', '914', '817', '309',
                    '331', '1307', '217', '1195', '1470', '1914', '525', '1560', '901', '1277', '200', '687', '648',
                    '1431', '219', '184', '415', '1471', '399', '768', '1983', '1615', '583', '1072', '953', '1060',
                    '779', '178', '1271', '792', '1331', '1621', '213', '335', '457', '865', '1303', '661', '386',
                    '637', '1838', '118', '1550', '1907', '1358', '427', '1644', '610', '1645', '989', '1845',
                    '514', '1763', '615', '389', '1052', '1711', '1512', '1487', '1532', '189', '561', '448', '343',
                    '1418', '1189', '138', '1041', '1534', '1102', '791', '1217', '1862', '1849', '506', '1989',
                    '1860', '654', '753', '942', '105', '1945', '148', '1700', '1180', '338', '406', '678', '86',
                    '390', '1619', '102', '11', '959', '80', '387', '508', '129', '1128', '1795', '1218', '31',
                    '1992', '214', '293', '636', '1709', '612', '581', '812', '1628', '1447', '1305', '1350',
                    '1059', '816', '226', '1190', '1411', '1640', '1982', '1020', '1312', '618', '1912', '968',
                    '851', '1362', '1533', '1976', '1633', '176', '223', '677', '621', '1611', '17', '769', '1158',
                    '270', '1110', '353', '613', '985', '365', '1642', '1643', '1169', '1714', '1674', '1724',
                    '268', '742', '125', '1761', '1365', '256', '883', '1284', '1240', '711', '81', '305', '1302',
                    '1670', '282', '672', '1906', '258', '1138', '794', '295', '1681', '442', '1988', '957', '333',
                    '1414', '1561', '378', '1951', '263', '312', '128', '1366', '1065', '464', '1405', '417',
                    '1199', '182', '1007', '460', '1920', '1909', '40', '977', '1823', '1607', '1877', '1298',
                    '1225', '782', '852', '1578', '1685', '1655', '1344', '1339', '596', '18', '1617', '691', '796',
                    '1579', '849', '1243', '424', '161', '945', '866', '1335', '1419', '899', '1098', '813', '1247',
                    '1231', '355', '882', '1062', '966', '8', '113', '1385', '1597', '1812', '1851', '893', '202',
                    '659', '325', '1586', '1242', '1863', '481', '1963', '1593', '955', '1682', '1555', '1675',
                    '743', '412', '511', '1973', '1574', '1731', '134', '1222', '197', '582', '747', '283', '368',
                    '1024', '1359', '1167', '1000', '1531', '385', '349', '149', '973', '746', '587', '1910',
                    '1905', '900', '630', '398', '1095', '611', '577', '1856', '179', '269', '1773', '192', '913',
                    '680', '1023', '822', '361', '162', '1438', '783', '1915', '598', '1552', '877', '1601', '576',
                    '1214', '762', '1141', '1673', '745', '452', '592', '758', '1935', '1187', '132', '1489', '89',
                    '87', '1831', '1070', '1752', '64', '1739', '1771', '655', '441', '474', '1959', '351', '542',
                    '472', '1786', '1744', '1077', '1553', '1238', '553', '1703', '713', '1255', '94', '714',
                    '1525', '1029', '1918', '998', '625', '543', '1241', '938', '503', '1805', '1074', '13', '36',
                    '629', '1516', '599', '1683', '218', '45', '982', '20', '939', '1784', '1527', '215', '1146',
                    '1116', '320', '224', '1500', '1677', '718', '588', '510', '1276', '1826', '1211', '801',
                    '1442', '35', '555', '1730', '65', '1855', '604', '1853', '1317', '727', '556', '1663', '1454',
                    '793', '806', '370', '1580', '732', '1147', '315', '1901', '1627', '260', '311', '1136', '965',
                    '1564', '1701', '1459', '990', '1149', '212', '1931', '1716', '1927', '38', '875', '140', '181',
                    '1822', '1306', '1182', '589', '1125', '664', '1296', '1127', '1103', '479', '1228', '329',
                    '735', '1441', '1398', '1144', '1839', '1213', '1085', '216', '321', '255', '272', '974',
                    '1764', '1112', '614', '1765', '332', '72', '830', '1373', '710', '1403', '1367', '1939',
                    '1742', '1201', '579', '1529', '1934', '1829', '278', '1760', '1455', '1842', '1192', '382',
                    '1309', '1726', '58', '1054', '1200', '1404', '1595', '1993', '183', '1899', '536', '1352',
                    '833', '1053', '347', '847', '1328', '19', '1609', '936', '1634', '1300', '416', '844', '898',
                    '995', '1508', '1421', '1160', '725', '1732', '656', '1868', '168', '1174', '1647', '1875',
                    '960', '438', '1778', '1018', '48', '632', '500', '1572', '623', '751', '1278', '195', '855',
                    '858', '524', '1940', '1383', '121', '736', '1575', '108', '145', '88', '1616', '1559', '1376',
                    '566', '1244', '425', '1523', '204', '568', '540', '1498', '237', '1413', '123', '1710', '439',
                    '153', '690', '1203', '892', '297', '228', '221', '146', '1902', '1444', '1968', '1542', '393',
                    '1777', '1031', '435', '30', '653', '1535', '1903', '688', '698', '401', '518', '601', '1941',
                    '803', '1632', '628', '951', '1375', '318', '733', '1626', '356', '1637', '1954', '1758',
                    '1770', '1898', '910', '1735', '937', '1576', '1798', '767', '1629', '397', '1232', '943', '71',
                    '520', '683', '1273', '241', '1291', '570', '1371', '1775', '1104', '728', '323', '631', '74',
                    '144', '823', '405', '461', '1715', '209', '185', '722', '703', '334', '1337', '104', '1734',
                    '426', '1753', '443', '1111', '313', '1800', '1063', '348', '1669', '1064', '1130', '774',
                    '1067', '1876', '1080', '1308', '1069', '819', '111', '266', '1093', '1028', '1916', '1015',
                    '1592', '1932', '1736', '1427', '1005', '512', '1299', '362', '1106', '1075', '290', '1034',
                    '1409', '1261', '1456', '34', '1636', '1952']
validation_user_id = ['1143', '1088', '12', '1961', '151', '1772', '1815', '473', '1445', '1318', '327', '517',
                      '314', '642', '1816', '685', '1864', '665', '403', '1363', '737', '239', '1658', '1193',
                      '1251', '538', '1073', '919', '493', '1509', '805', '211', '1536', '498', '47', '1848',
                      '1466', '720', '1408', '1847', '1504', '597', '352', '906', '836', '1285', '999', '1785',
                      '172', '66', '1603', '547', '122', '1582', '126', '529', '1340', '230', '1348', '360', '933',
                      '1381', '528', '1384', '488', '1057', '1833', '1382', '1756', '1545', '502', '109', '1746',
                      '1987', '1117', '117', '1430', '1039', '1958', '42', '345', '781', '835', '1547', '1866',
                      '301', '1461', '1936', '1819', '1424', '1022', '701', '1846', '670', '346', '1030', '379',
                      '155', '73', '1563', '848', '1262', '853', '760', '98', '1433', '1486', '1108', '1429', '522',
                      '1319', '571', '394', '1462', '1224', '27', '1370', '788', '1843', '861', '1329', '1071',
                      '234', '9', '1196', '640', '831', '686', '1966', '1260', '135', '483', '1463', '862', '1068',
                      '275', '730', '233', '1900', '1326', '867', '1943', '1499', '1944', '1349', '499', '560',
                      '532', '37', '1107', '807', '620', '818', '359', '1379', '1288', '921', '245', '1170', '660',
                      '617', '1396', '1517', '1010', '463', '1101', '41', '1750', '453', '339', '449', '1488',
                      '1696', '749', '1620', '523', '1249', '1551', '1440', '1166', '748', '1921', '1013', '644',
                      '1482', '1474', '1315', '1521', '1698', '1870', '689', '744', '432', '1522', '772', '252',
                      '446', '277', '1209', '1446', '1361', '585', '676', '911', '1248', '1356', '726', '1887',
                      '70', '1558', '673', '1208', '456', '559', '1895', '273', '249', '695', '840', '186', '1608',
                      '358', '1460', '392', '643', '1880', '1202', '1420', '1021', '165', '1520', '1351', '1183',
                      '826', '752', '705', '413', '130', '152', '1083', '1539', '340', '46', '785', '539', '5',
                      '431', '1448', '1590', '1751', '1661', '147', '984', '480', '1058', '1267', '407', '1865',
                      '1066', '369', '322', '1767', '1082', '69', '1275', '1055', '696', '1084', '1953', '1692',
                      '1588', '422', '975', '1139', '1323', '1484', '1392', '1799', '1594', '489', '364', '444',
                      '1600', '120', '693', '1791', '95', '1820', '1568', '1173', '894', '964', '1985', '741',
                      '1485', '1824', '1549', '1888', '137', '433', '1330', '232', '451', '1428', '917', '925',
                      '1229', '16', '928', '1780', '578', '1003', '1394', '1236', '1492', '1124', '704', '291',
                      '1740', '468', '991', '1506', '1100', '825', '724', '1836', '870', '765', '1097', '616',
                      '288', '567', '79', '7', '997', '1859', '573', '495', '1759', '367', '1145', '391', '388',
                      '1852', '1969', '363', '1787', '421', '6', '1671', '1832', '1118', '1137', '1345', '880',
                      '979', '1513', '156', '1165', '1797', '372', '373', '1321', '1047', '1577', '131', '82',
                      '380', '1391', '1984', '634', '1026', '814', '787', '14', '922', '54', '1624', '777', '166',
                      '1437', '1705', '607', '551', '1793', '1205', '828', '537', '1892', '420', '10', '77', '1925',
                      '1049', '115', '717', '609', '1435', '486', '702', '1947', '1530', '1964', '721', '53',
                      '1221', '1397', '854', '700', '800', '1904', '67', '1176', '1861', '1458', '859', '1858',
                      '1704', '981', '1965', '57', '658', '1613', '1009', '289', '1960', '164', '1919', '26', '29',
                      '1693', '1680', '1207', '1612', '967', '1204', '1974', '1540', '1387', '1922', '972', '1393',
                      '699', '1230', '868', '1338', '600', '530', '515', '526', '1119', '400', '935', '281', '1821',
                      '1694', '491', '497', '544', '1507', '436', '694', '222', '944', '1501', '298', '1036',
                      '1811', '267', '863', '1584', '1389', '141', '1178', '1690', '622', '1237', '377', '1235',
                      '961', '766', '1332', '798', '1129', '639', '319', '1747', '246', '1569', '1894', '1334',
                      '76', '1422', '707', '262', '1056', '1177', '23', '1713']
test_user_id = ['1', '2', '3', '4', '21', '22', '24', '25', '32', '33', '43', '44', '50', '51', '55', '56', '60',
                '61', '75', '78', '83', '84', '85', '91', '92', '93', '103', '110', '112', '114', '116', '124',
                '133', '143', '157', '158', '159', '160', '170', '173', '174', '177', '180', '187', '191', '193',
                '194', '198', '201', '203', '205', '208', '210', '225', '227', '229', '236', '238', '243', '244',
                '247', '248', '253', '254', '261', '264', '271', '274', '276', '280', '285', '286', '287', '302',
                '303', '304', '307', '308', '324', '326', '328', '336', '341', '342', '344', '350', '357', '371',
                '374', '375', '381', '383', '396', '402', '404', '408', '411', '414', '423', '428', '429', '434',
                '440', '450', '454', '455', '459', '462', '465', '466', '469', '470', '471', '475', '487', '490',
                '496', '501', '504', '505', '513', '519', '521', '527', '531', '534', '535', '541', '545', '546',
                '548', '549', '550', '552', '554', '557', '564', '565', '574', '575', '584', '586', '590', '591',
                '594', '595', '602', '603', '606', '608', '619', '627', '633', '646', '647', '649', '650', '652',
                '662', '663', '666', '669', '674', '679', '681', '684', '697', '709', '715', '716', '719', '723',
                '731', '738', '739', '740', '756', '757', '759', '761', '763', '764', '770', '773', '775', '776',
                '778', '780', '784', '786', '789', '790', '797', '799', '802', '809', '811', '815', '821', '824',
                '832', '834', '838', '839', '843', '846', '850', '856', '871', '872', '873', '878', '879', '881',
                '886', '888', '891', '896', '897', '902', '903', '907', '908', '912', '915', '916', '918', '924',
                '927', '929', '930', '931', '932', '940', '946', '947', '949', '956', '958', '963', '969', '970',
                '971', '980', '983', '996', '1001', '1002', '1004', '1006', '1011', '1012', '1014', '1016', '1017',
                '1025', '1027', '1032', '1038', '1040', '1044', '1046', '1061', '1076', '1078', '1081', '1086',
                '1090', '1092', '1096', '1099', '1105', '1109', '1115', '1120', '1122', '1123', '1126', '1133',
                '1135', '1140', '1142', '1148', '1150', '1151', '1152', '1154', '1156', '1159', '1161', '1162',
                '1168', '1171', '1172', '1181', '1184', '1185', '1186', '1188', '1194', '1210', '1219', '1227',
                '1233', '1239', '1245', '1246', '1250', '1253', '1256', '1257', '1258', '1263', '1266', '1272',
                '1274', '1279', '1280', '1283', '1287', '1289', '1290', '1292', '1293', '1294', '1295', '1297',
                '1301', '1304', '1310', '1311', '1313', '1316', '1333', '1336', '1343', '1346', '1354', '1355',
                '1357', '1360', '1368', '1377', '1378', '1386', '1388', '1395', '1400', '1401', '1402', '1410',
                '1416', '1425', '1426', '1432', '1434', '1436', '1439', '1443', '1449', '1451', '1452', '1453',
                '1457', '1464', '1465', '1467', '1469', '1472', '1475', '1476', '1480', '1481', '1490', '1493',
                '1494', '1496', '1497', '1503', '1511', '1514', '1519', '1528', '1537', '1538', '1544', '1546',
                '1554', '1562', '1565', '1571', '1581', '1583', '1585', '1587', '1589', '1596', '1598', '1605',
                '1606', '1610', '1618', '1622', '1623', '1630', '1631', '1639', '1641', '1646', '1648', '1650',
                '1651', '1656', '1662', '1664', '1665', '1666', '1667', '1668', '1676', '1679', '1686', '1687',
                '1688', '1689', '1695', '1697', '1706', '1708', '1712', '1717', '1718', '1719', '1720', '1721',
                '1725', '1727', '1728', '1738', '1741', '1743', '1754', '1766', '1768', '1769', '1774', '1779',
                '1781', '1782', '1789', '1790', '1792', '1803', '1804', '1808', '1810', '1813', '1814', '1825',
                '1827', '1828', '1850', '1854', '1857', '1871', '1872', '1882', '1883', '1884', '1886', '1893',
                '1896', '1897', '1908', '1911', '1913', '1917', '1923', '1928', '1930', '1933', '1937', '1938',
                '1942', '1946', '1948', '1956', '1957', '1962', '1967', '1970', '1971', '1972', '1978', '1979',
                '1981', '1986', '1990', '1996']
"""
training_user_id_watches_two_groups = [524, 754, 537, 854, 529, 520, 605, 848, 525, 719, 825, 809, 510, 606, 687, 806, 452, 607,
                            493, 588, 823, 623, 762, 536, 612, 603, 517, 744, 563, 581, 624, 371, 530, 650, 370, 498,
                            713, 721, 857, 618, 671, 831, 540, 822, 590, 679, 582, 579, 760, 689, 658, 850, 633, 655,
                            697, 765, 495, 816, 804, 637, 682, 635, 599, 630, 701, 805, 672, 598, 832, 162, 617, 545,
                            675, 827, 768, 654, 800, 601, 778, 759, 586, 508, 653, 80]
validation_user_id_watches_two_groups = [753, 670, 509, 532, 496, 758, 840, 48, 851, 813, 841, 711, 584, 788, 856, 795, 833, 589,
                              292, 712, 803, 737, 506, 627, 363, 702, 640, 834, 539, 665, 621, 731, 796, 839, 695, 638,
                              787, 79, 398, 514, 764, 538]
training_user_id_watches_five_groups = [794, 681, 508, 649, 529, 712, 620, 699, 761, 526, 498, 715, 586, 564, 671, 825,
                                        497, 802, 507, 520, 596, 503, 751, 820, 370, 688, 515, 695, 560, 822, 578, 363,
                                        572, 585, 658, 506, 740, 733, 765, 525, 837, 713, 500, 767, 644, 533, 581, 754,
                                        823, 696, 674, 824, 737, 781, 636, 690, 608, 651, 795, 367, 601, 742, 702, 789,
                                        563, 720, 613, 721, 775, 808, 652, 452, 687, 689, 557, 849, 602, 656, 748, 835,
                                        545, 807, 759, 755, 729, 552, 809, 607, 518, 590, 853, 513, 555, 657, 785, 832,
                                        38, 811, 244, 598, 677, 768, 777, 541, 522, 670, 782, 571, 553, 510, 667, 594,
                                        493, 562, 629, 805, 693, 845, 783, 592, 530, 499, 618, 852, 531, 800, 666, 599,
                                        723, 704, 519, 547, 846, 685, 747, 589, 842, 861, 605, 796, 626, 625, 645, 764,
                                        774, 603, 617, 745, 847, 719, 856, 786, 542, 678, 539, 703, 717, 548, 502, 593,
                                        514, 843, 243, 697, 512, 574, 623, 700, 730, 621, 710, 749, 813, 812, 828, 857,
                                        814, 750, 619, 524, 648, 606, 212, 511, 650, 559, 718, 778, 633, 827, 398, 580,
                                        292, 711, 821]
validation_user_id_watches_five_groups = [641, 565, 804, 394, 582, 691, 746, 826, 757, 758, 819, 791, 162, 854, 643,
                                          744, 772, 851, 738, 763, 694, 815, 676, 787, 725, 352, 642, 532, 443, 475,
                                          669, 728, 600, 135, 635, 858, 701, 573, 762, 790, 739, 743, 639, 848, 611,
                                          495, 727, 534, 551, 831, 665, 588, 632, 705, 722, 583, 604, 655, 680, 683,
                                          724, 535, 732, 803, 579, 566, 816, 829, 792, 527, 615, 536, 663, 756, 494,
                                          818, 653, 544, 675, 79, 587, 776, 616, 770, 610, 622, 549, 597, 646, 661,
                                          773, 752, 396, 538, 801, 760, 855, 769]


with open('data/final_corpus_dictionary_max_length_' + str(max_review_length) + '.pickle', 'rb') as handle:
    corpus = pickle.load(handle)
corpus = transform_age_category_5(corpus, topic)
posts_id_to_del = []
for i in range(len(corpus)):
    if corpus[i]["age"] == "to_del":
        posts_id_to_del.append(i)
    elif len(corpus[i]["text_sequence"]) == 0:
        posts_id_to_del.append(i)
for index in sorted(posts_id_to_del, reverse=True):
    del corpus[index]

training = []
validation = []
test = []
training_user_id = []
validation_user_id = []
if topic == "Watches" and num_categories == 2:
    training_user_id = training_user_id_watches_two_groups
    validation_user_id = validation_user_id_watches_two_groups
elif topic == "Watches" and num_categories == 5:
    training_user_id = training_user_id_watches_five_groups
    validation_user_id = validation_user_id_watches_five_groups
for message_dict in corpus:
    user_id = int(message_dict["user_id"])
    if user_id in training_user_id:
        training.append(message_dict)
    elif user_id in validation_user_id:
        validation.append(message_dict)
    else:
        test.append(message_dict)
        
X_train, y_train = create_x_y(training, task)
X_train = pad_sequences(X_train, maxlen=max_review_length, padding='post')
X_validation, y_validation = create_x_y(validation, task)
X_validation = pad_sequences(X_validation, maxlen=max_review_length, padding='post')
X_test, y_test = create_x_y(test, task)
X_test = pad_sequences(X_test, maxlen=max_review_length, padding='post')

if word_embedding_dict == "itwac":
    with open('data/itwac_word_embedding_matrix.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)
else:
    with open('data/twitter_word_embedding_matrix.pickle', 'rb') as handle:
        embedding_matrix = pickle.load(handle)
"""
with open(filePathMainInfoTrain, "w") as f:
    if topic == "Watches" and num_categories == 2:
        num_0 = 0
        num_1 = 0
        for post in corpus:
            if post["age"] == 0:
                num_0 += 1
            if post["age"] == 1:
                num_1 += 1
        f.write("<30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(corpus))) + "%\n")
        f.write(">49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(corpus))) + "%\n")
        num_0 = 0
        num_1 = 0
        for post in training:
            if post["age"] == 0:
                num_0 += 1
            if post["age"] == 1:
                num_1 += 1
        f.write("training <30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(training))) + "%\n")
        f.write("training >49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(training))) + "%\n")
        num_0 = 0
        num_1 = 0
        for post in validation:
            if post["age"] == 0:
                num_0 += 1
            if post["age"] == 1:
                num_1 += 1
        f.write("validation <30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(validation))) + "%\n")
        f.write("validation >49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(validation))) + "%\n")
        num_0 = 0
        num_1 = 0
        for post in test:
            if post["age"] == 0:
                num_0 += 1
            if post["age"] == 1:
                num_1 += 1
        f.write("test <30    : " + str(num_0) + "; percent: " + str(round(num_0 * 100.0 / len(test))) + "%\n")
        f.write("test >49,<70: " + str(num_1) + "; percent: " + str(round(num_1 * 100.0 / len(test))) + "%\n")
    elif topic == "Watches" and num_categories == 5:
        arr_categories = [0, 0, 0, 0, 0]
        arr_labels = ["20-29", "30-39", "40-49", "50-59", "60-69"]
        for post in corpus:
            for i in range(len(arr_categories)):
                arr_categories[i] += post["age"][i]
        for i in range(len(arr_categories)):
            f.write(arr_labels[i] + ": " + str(arr_categories[i]) + "; percent: " + str(
                round(arr_categories[i] * 100.0 / len(corpus))) + "%\n")

        arr_categories = [0, 0, 0, 0, 0]
        for post in training:
            for i in range(len(arr_categories)):
                arr_categories[i] += post["age"][i]
        for i in range(len(arr_categories)):
            f.write("training " + arr_labels[i] + ": " + str(arr_categories[i]) + "; percent: " + str(
                round(arr_categories[i] * 100.0 / len(training))) + "%\n")

        arr_categories = [0, 0, 0, 0, 0]
        for post in validation:
            for i in range(len(arr_categories)):
                arr_categories[i] += post["age"][i]
        for i in range(len(arr_categories)):
            f.write("validation " + arr_labels[i] + ": " + str(arr_categories[i]) + "; percent: " + str(
                round(arr_categories[i] * 100.0 / len(validation))) + "%\n")

        arr_categories = [0, 0, 0, 0, 0]
        for post in test:
            for i in range(len(arr_categories)):
                arr_categories[i] += post["age"][i]
        for i in range(len(arr_categories)):
            f.write("test " + arr_labels[i] + ": " + str(arr_categories[i]) + "; percent: " + str(
                round(arr_categories[i] * 100.0 / len(test))) + "%\n")
    f.flush()
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nSimple LSTM\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_simple.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name, actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_dropout_1.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_dropout_2.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/lstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = lstm.classification.lstm_with_cnn.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nSimple BLSTM\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_simple_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_simple.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nBLSTM Dropout 1\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_dropout_1_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_dropout_1.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()

with open(filePathMainInfoTrain, "a") as f:
    f.write("\nBLSTM Dropout 2\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_dropout_2_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_dropout_2.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()
"""
numNeurons = [50, 100, 129]
with open(filePathMainInfoTrain, "a") as f:
    f.write("\nBLSTM with CNN\n")
    for n in numNeurons:
        save_model_name = "models/" + str(task) + "/" + str(word_embedding_dict) + "/" + str(max_review_length) + "/blstm_with_cnn_neurons_" + str(n) + "_attempt_0"
        f.write("\n\t" + str(n) + " neurons" + "\n")
        for i in range(repeat):
            save_model_name = save_model_name[:-1] + str(i+1)
            actual_epochs = blstm.classification.blstm_with_cnn.runTraining(X_train, y_train, X_validation, y_validation, embedding_matrix, n, num_categories, batch_size, early_stopping_wait, save_model_name)
            f.write(
                resultsOfTrainingToFile(X_train, y_train, X_validation, y_validation, save_model_name,
                                        actual_epochs))
            f.write(resultsOfTest(X_test, y_test, save_model_name))
            f.flush()
    f.write("\n")
    f.close()
