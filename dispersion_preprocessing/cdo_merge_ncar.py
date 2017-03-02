from operator import attrgetter
from argparse import ArgumentParser
from subprocess import call

if __name__ == '__main__':
    parser = ArgumentParser(description='Extract variables from netcdf file')
    parser.add_argument('-ipl', '--inputpl',required=True,type=str,
                        help='pressure level filename')
    parser.add_argument('-isfc', '--inputsfc',required=True,type=str,
                        help='surface level filename')
    opts = parser.parse_args()
    getter = attrgetter('inputpl','inputsfc')
    inpl,insfc = getter(opts)
    plinputpath = ''
    sfcinputpath = ''
    fpl = open(inpl,'r')
    sfc = open(insfc,'r')
    for pos,line in enumerate(sfc):
        strsc = line.strip()
        struv = line.replace("sc.","uv.").strip()
        name = line.split('.')
        for line2 in sfc:
            name2 = line2.split('.')
            if name2[5] == name[5]:
               strsfc = line2.strip()
               break
        print strsc,struv,strsfc
        #stdout = open(name[5],"wb")
        call(["cdo","merge",plinputpath+"/"+strsc,plinputpath+"/"+struv,sfcinputpath+"/"+strsfc,str(name[5]).strip()])
        #call(["cat",plinputpath+"/"+strsc,plinputpath+"/"+struv,sfcinputpath+"/"+strsfc],stdout=stdout)
    fpl.close()
    sfc.close()
