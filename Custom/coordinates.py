from core.config import cfg
from geopy.distance import geodesic
from geopy import Point
import math



def getCoordinatesFromDistanceAngle(distmeters,bearing,lat1,lon1):
    # given: lat1, lon1, bearing, distMeters
    ans= geodesic(meters=distmeters).destination(Point(lat1, lon1), bearing)
    return ans.latitude,ans.longitude


def angle3(a, b, c):
    ang = math.degrees(math.atan2(c[1]-b[1], c[0]-b[0]) - math.atan2(a[1]-b[1], a[0]-b[0]))
    return ang + 360 if ang < 0 else ang



def FramesToCoordinatesAndDistance(OriginCoordinates,OriginFrame,DestinationFrame,presentAltitude,Frame):
    dici={}

    xi,yi=OriginFrame
    xf,yf=DestinationFrame
    xic,yic=OriginCoordinates
    xframe,yframe=Frame

    dist_x_meters=(xf-xi)*(cfg.Coordinates.x_width*presentAltitude/(cfg.Coordinates.height*xframe))
    dist_y_meters=(yi-yf)*(cfg.Coordinates.y_width*presentAltitude/(cfg.Coordinates.height*yframe))

    bearing=angle3([OriginFrame[0],0],OriginFrame,DestinationFrame)

    dist=((dist_x_meters**2)+(dist_y_meters**2))**(0.5)

    newLatitude,newLongitude=getCoordinatesFromDistanceAngle(dist,bearing,xic,yic)

    dici["bearing"]=360-bearing
    dici["dist_x_meters"]=bearing
    dici["dist_x_meters"]=dist_x_meters
    dici["dist_y_meters"]=dist_y_meters
    dici["newLongitude"]=newLongitude
    dici["newLatitude"]=newLatitude
    dici["dist"]=dist
    print(dici)
    return dici

# print(FramesToCoordinatesAndDistance([10.03,9.56],[45,7],[8,9],90,[100,69]))
