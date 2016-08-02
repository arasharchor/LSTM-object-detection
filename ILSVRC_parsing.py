import xml.etree.ElementTree as ET

def parse_ILSVRCXML(path):
    tree = ET.parse(path)
    root = tree.getroot()

    folder = root.find('folder').text
    filename = root.find('filename').text
    database = root.find('source').find('database').text

    width = root.find('size').find('width').text
    height = root.find('size').find('height').text

    objects = []

    for obj in root.findall('object'):
        trackid = obj.find('trackid').text
        name = obj.find('name').text
        xmax = obj.find('bndbox').find('xmax').text
        xmin = obj.find('bndbox').find('xmin').text
        ymax = obj.find('bndbox').find('ymax').text
        ymin = obj.find('bndbox').find('ymin').text
        occluded = obj.find('occluded').text
        generated = obj.find('generated').text
        objects.append([trackid, name, int(xmax), int(xmin), int(ymax), int(ymin), int(occluded), int(generated)])

    return folder, filename, database, int(width), int(height), objects

if __name__ == "__main__":
    print parse_ILSVRCXML('xml/000001.xml')
