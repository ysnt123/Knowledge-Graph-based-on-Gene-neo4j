import re
from py2neo import Graph,Node,Relationship,NodeMatcher
import numpy as np
from scipy import spatial
import gensim
import jieba
import numpy as np
from scipy.linalg import norm

"""
函数说明：输入一个字符串，返回图查询的结果

"""
def get_Answer(string):
    graph = Graph(host="host",auth=("neo4j","neo4j"))
    data1 = graph.run(string).to_data_frame()
    return data1





    '''
函数说明：
用于语句计算相似度的函数
'''
model_file = 'GoogleNews-vectors-negative300.bin'
model = gensim.models.KeyedVectors.load_word2vec_format(model_file, binary=True)
index2word_set = set(model.wv.index2word)

def avg_feature_vector(sentence, model, num_features, index2word_set):
    words = sentence.split()
    feature_vec = np.zeros((num_features, ), dtype='float32')
    n_words = 0
    for word in words:
        if word in index2word_set:
            n_words += 1
            feature_vec = np.add(feature_vec, model[word])
    if (n_words > 0):
        feature_vec = np.divide(feature_vec, n_words)
    return feature_vec
    
    
    
    
    
    
    '''
函数说明：导入所有问题模板
输入："question_pattern.txt"
输出：所有问题组成的list
'''
def qPattern(a):
    qList=[]
    with open(a, 'r') as file:
        for l in  file:
            qList.append(l.strip('\n'))
    return(qList)




'''
问题预处理，将一些无法被识别的sequence变成id，增强对问题类型的识别能力
protein:{id}   Q03073
Species{name}
Gene:{id: 'AET4Gv20696400', species: 'Aegilops tauschii'})
GO:{id: 'GO:1902494'})(_204455104:GO {id: 'GO:0004553'}
Sequence:UPI00013ED01D
'''
def realQ(Question):
    verb=Question.split()
    for i in verb:
        if str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').isalpha():
            continue
        elif str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').isalnum():
            if str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').isdigit():
                a=verb.index(i)
                verb[a]='GO id'
            elif str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').startswith(('AET','AMTR','g','Al','fgenesh1','fgenesh2','scaffold','AT','BVRB','BRADI','GSBRNA2T','Bo','Bra','CHLRE','CHC','CCACVL','Csa','CM','DCAR','Dr','Gasu','GLYMA','B456','HannXRQ','HORVU','LPERR','TanjilG','MANES','MTR','GSMUA','A4A49','OBART','OB','ORGLA','OGLUM','KN','AMD','OMERI','ONIVA','OPUNC','ORUFI','BGIOSGA','Os','OSTLU','PHAVU','Pp','POPTR','PRUPE','SELMODRAFT','SETIT','Solyc','PGS','SORBI')):
                a=verb.index(i)
                verb[a]='Gene id'
            elif str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').startswith('UPI'):
                a=verb.index(i)
                verb[a]='Sequence id'
            else:
                a=verb.index(i)
                verb[a]='Protein id'
    realQ=' '.join(verb)
    realQ=realQ.strip('{').strip('}').strip('(').strip(')')
    return(realQ)
    
    
    '''
函数说明：将问题和question_pattern计算相似度
输入：question
{AET4Gv20696400} comes from how many (Species)?
what (genes) do {Aegilops tauschii} has?
Gene Os01g0740400 is transcribed into what sequence?
sequence UPI00004C2817 is transcribed from which gene?
输出：相似度最大的问题的类型
'''
#Q='sequence UPI00004C2817 is transcribed from which gene?'
#Question=realQ(Q)
#qpattern_dir="question_pattern.txt"
def max_sim(Question,qpattern_dir):
    m=qPattern(qpattern_dir)
    obpattern = r'[\]](.*?)[?]'
    s0_afv = avg_feature_vector(Question, model=model, num_features=300, index2word_set=index2word_set)
    simList=[]
    for i in m:
        text = i
        patternObj = re.compile(obpattern)
        result1 = str(patternObj.findall(text)).lstrip('[\'').rstrip('\']')
        s1_afv = avg_feature_vector(result1, model=model, num_features=300, index2word_set=index2word_set)
        sim = 1 - spatial.distance.cosine(s0_afv, s1_afv)
        simList.append(sim)
    #print(simList)  #查看相似度的数值
    max_index=simList.index(max(simList))
    text2=m[max_index]
    pattern1 = r'[\[](.*?)[\]]'
    patternObj2 = re.compile(pattern1)
    result2 = patternObj2.findall(text2)
    return(str(result2).lstrip('[\'').rstrip('\']'))
#max_sim(Question,qpattern_dir)


'''
函数目标：
输入问题，提取里面的实体，宽泛点~
.啊啊啊啊
protein:{id}   Q03073
Species{name}
Gene:{id: 'AET4Gv20696400', species: 'Aegilops tauschii'})
GO:{id: 'GO:1902494'})(_204455104:GO {id: 'GO:0004553'}
Sequence:UPI00013ED01D
{UPI4Gv20696400} {comes from} how many (Species)?
'''

class entity_extract():
    def __init__(self,question):
        self.question=question
    def get_id(self):
        dictt=dict()
        verb=self.question
        verb=verb.split()
        for i in verb:
            if str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').rstrip('\?').isalpha():
                continue
            elif str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').rstrip('\?').isalnum():
                if str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').rstrip('\?').isdigit():
                    a=verb.index(i)
                    dictt['GO id']=verb[a]
                    return(dictt)
                elif str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').rstrip('\?').startswith(('AET','AMTR','g','Al','fgenesh1','fgenesh2','scaffold','AT','BVRB','BRADI','GSBRNA2T','Bo','Bra','CHLRE','CHC','CCACVL','Csa','CM','DCAR','Dr','Gasu','GLYMA','B456','HannXRQ','HORVU','LPERR','TanjilG','MANES','MTR','GSMUA','A4A49','OBART','OB','ORGLA','OGLUM','KN','AMD','OMERI','ONIVA','OPUNC','ORUFI','BGIOSGA','Os','OSTLU','PHAVU','Pp','POPTR','PRUPE','SELMODRAFT','SETIT','Solyc','PGS','SORBI')):
                    a=verb.index(i)
                    dictt['Gene id']=verb[a]
                    return(dictt)
                elif str(i).lstrip('[\'').rstrip('\']').strip('{').strip('}').strip('(').strip(')').rstrip('\?').startswith('UPI'):
                    a=verb.index(i)
                    dictt['Sequence id']=verb[a]
                    return(dictt)
                else:
                    a=verb.index(i)
                    dictt['Protein id']=verb[a]
                    return(dictt)
    def get_species(self):
        verb=self.question
        species_type=["Aegilops tauschii","Arabidopsis halleri","Arabidopsis lyrata","Amborella trichopoda","Beta vulgaris","Brassica rapa","Chondrus crispus","Corchorus capsularis","Cyanidioschyzon","Daucus carota","Dioscorea rotundata","Galdieria sulphuraria","Gossypium raimondii","Hordeum vulgare","Oryza brachyantha","Oryza glaberrima","Oryza glumipatula","Arabidopsis thaliana","Brachypodium distachyon","Brassica napus","Brassica oleracea","Chlamydomonas reinhardtii","Cucumis sativus","Glycine max","Helianthus annuus","Leersia perrieri","Lupinus angustifolius","Manihot esculenta","Medicago truncatula","Musa acuminata","Nicotiana attenuata","Oryza barthii","Oryza longistaminata","Oryza rufipogon","Oryza sativa Indica Group","Phaseolus vulgaris","Populus trichocarpa","Oryza meridionalis","Oryza nivara","Oryza punctata","Oryza sativa Japonica Group","Ostreococcus lucimarinus","Physcomitrella patens","Selaginella moellendorffii","Setaria italica","Solanum tuberosum","Theobroma cacao","Vigna angularis","Vigna radiata","Zea mays","Prunus persica","Solanum lycopersicum","Sorghum bicolor","Trifolium pratense","Triticum aestivum","Triticum dicoccoides","Triticum urartu","Vitis vinifera"]
        for i in species_type:
            if i in verb:
                return (i)
            else:
                '''
                请提供一个精度较高的模块呀- -
                '''
                continue
#m=entity_extract('what genes do molecular function includes in Vitis vinifera ?')
#print(m.get_species())
#print('Vitis vinifera' in 'what genes do molecular function includes in Vitis vinifera ?')




def AnswerQ(string,Question,num=10):
    if string=='have1':
        body=entity_extract(Question)
        species_type=str(body.get_species())
        a='match(na:Species{{name:"{species_type}"}})-[have]->(nb:Gene) return nb.id limit {num}'.format(species_type=species_type,num=str(num))
        return(a)
    elif string=='have2':
        #[have2] What species do gene id exist in?
        body=entity_extract(Question)
        ids=body.get_id()['Gene id']
        a='match(na:Species)-[have]->(nb:Gene{{id:"{geneid}"}}) return na.name limit {num}'.format(geneid=ids,num=str(num))
        return(a)
    elif string=='be_transcribed_into1':
        # [be_transcribed_into1] Gene id is transcribed into what sequence?
        body=entity_extract(Question)
        ids=body.get_id()['Gene id']
        a='match(na:Gene{{id:"{geneid}"}})-[be_transcribed_into]->(nb:Sequence) return nb.id limit {num}'.format(geneid=ids,num=str(num))
        return(a)
    elif string=='be_transcribed_into2':
        #[be_transcribed_into2] Gene id is transcribed into what protein?
        body=entity_extract(Question)
        ids=body.get_id()['Gene id']
        a='match(na:Gene{{id:"{geneid}"}})-[be_transcribed_into]->(nb:Protein) return nb.id,nb.name limit {num}'.format(geneid=ids,num=str(num))
        return(a)
    elif string=='be_transcribed_into32':
        #[be_transcribed_into3] sequence/protein id is transcribed from which gene?
        body=entity_extract(Question)       
        ids=body.get_id()['Protein id']
        a='match(na:Gene)-[be_transcribed_into]->(nb:Protein{{id:"{pid}"}}) return na.id limit {num}'.format(pid=ids,num=str(num))
        return(a)
    elif string=='be_transcribed_into31':
        body=entity_extract(Question)
        ids=body.get_id()['Sequence id']
        a='match(na:Gene)-[be_transcribed_into]->(nb:Sequence{{id:"{sid}"}}) return na.id limit {num}'.format(sid=ids,num=str(num))
        return(a)
    elif string=='is_a':
        #[is_a] what is the GO that has relationship "is a " with GO id?
        body=entity_extract(Question)
        ids=body.get_id()['GO id']
        a='match(na:GO{{id:"GO:{gid}"}})-[is_a]->(nb:GO) return nb.id limit {num}'.format(gid=ids,num=str(num))
        return(a)
    elif string=='negatively_regulates':
        #[negatively_regulates] what is the GO that has relationship "negatively regulates " with GO id?
        body=entity_extract(Question)
        ids=body.get_id()['GO id']
        a='match(na:GO{{id:"GO:{gid}"}})-[negatively_regulates]->(nb:GO) return nb.id limit {num}'.format(gid=ids,num=str(num))
        return(a)
    elif string=='positively_regulates':
        #[positively_regulates] what is the GO that has relationship "positively regulates " with GO id?
        body=entity_extract(Question)
        ids=body.get_id()['GO id']
        a='match(na:GO{{id:"GO:{gid}"}})-[positively_regulates]->(nb:GO) return nb.id limit {num}'.format(gid=ids,num=str(num))
        return(a)
    elif string=='regulates':
        #[regulates] what is the GO that has relationship "regulates " with GO id?
        body=entity_extract(Question)
        ids=body.get_id()['GO id']
        a='match(na:GO{{id:"GO:{gid}"}})-[regulates]->(nb:GO) return nb.id limit {num}'.format(gid=ids,num=str(num))
        return(a)
    elif string=='part_of':
        #[part_of] what is the GO that has relationship "part of" with GO id?
        body=entity_extract(Question)
        ids=body.get_id()['GO id']
        a='match(na:GO{{id:"GO:{gid}"}})-[part_of]->(nb:GO) return nb.id limit {num}'.format(gid=ids,num=str(num))
        return(a)
    elif string=='molecular_function':
        #[molecular_function] what genes do molecular function includes in Species?
        body=entity_extract(Question)
        species_type=str(body.get_species())
        a='match(na:Gene{{species:"{species_type}"}})-[molecular_function]->(nb:GO) return na.id,na.species limit {num}'.format(species_type=species_type,num=str(num))
        return(a)
    elif string=='biological_process':
        #[biological_process] what genes do biological process includes?
        body=entity_extract(Question)
        species_type=str(body.get_species())
        a='match(na:Gene{{species:"{species_type}"}})-[biological_process]->(nb:GO) return na.id,na.species limit {num}'.format(species_type=species_type,num=str(num))
        return(a)
    elif string=='cellular_component':
        #[cellular_component] what genes do cellular component includes?
        body=entity_extract(Question)
        species_type=str(body.get_species())
        a='match(na:Gene{{species:"{species_type}"}})-[cellular_component]->(nb:GO) return na.id,na.species limit {num}'.format(species_type=species_type,num=str(num))
        return(a)
    elif string=='eco':
        #[eco] what genes is an evidence ontology in Species?
        body=entity_extract(Question)
        species_type=str(body.get_species())
        a='match(na:Gene{{species:"{species_type}"}})-[eco]->(nb:GO) return na.id,na.species limit {num}'.format(species_type=species_type,num=str(num))
        return(a)
    elif string=='belong_to':
        #[belong_to] Sequence id belongs to which protein? 
        body=entity_extract(Question)
        ids=body.get_id()['Sequence id']
        a='match(na:Sequence{{id:"{sid}"}})-[belong_to]->(nb:Protein) return nb.id limit {num}'.format(sid=ids,num=str(num))
        return(a)
        
        
 '''
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
BOSS函数！！！！！
%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%


示例Q='{Aegilops tauschii} has how many genes?'
（因为只有这个句子算是写好了）

'''
def run():
    Q=input('Enter your Question:')
    Question=realQ(Q)
    qpattern_dir="question_pattern.txt"
    m=max_sim(Question,qpattern_dir)
    verb=AnswerQ(m,Q)# [,num=5000]
    print(get_Answer(verb))
    




'''
例子：{Aegilops tauschii} {have} how many (Genes)?
What species do gene AET4Gv20696400 exist in?
Gene Os01g0740400 is transcribed into what sequence?
Gene id is transcribed into what protein?
sequence UPI00004C2817 is transcribed from which gene?
what is the GO that has relationship "is a " with GO 0060255 ?
what is the GO that has relationship "negatively regulates " with GO 0000086 ?
what genes do molecular function includes in Vitis vinifera?
what genes is an evidence ontology in Sorghum bicolor?
Sequence UPI0000000444 belongs to which protein? 
运行函数

'''
if __name__=="__main__":
    run()   
        
