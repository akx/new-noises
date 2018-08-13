new-noises
==========

A char-rnn to generate

* new nonsensical genre names based on everynoise data.
* new IKEA product lines.

(It's more fun when it hasn't fully trained itself.)

Also an useful char-rnn template with full serialization/deserialization.

## Usage

### Training

* Prepare a dataset (or use `genres.txt` or `ikea.txt`)
* Run `python new_noises.py train --input-file=genres.txt --avoid-reality`.  
  (`--avoid-reality` ensures no items present in the original data are printed. You may leave it out.)
* Watch the output, copy-and-paste the most hilarious entries as you like.

### Sampling

* Once you have a .hdf5 file generated by training, you can reuse it with:  
  `python new_noises.py sample --model-file=genres.txt --avoid-reality`


## Genre Examples

```
abft indie ameriegaze hip hop
acgicary groove eddeo reggaetone
aid criac austro-indus
alternutin slassic ridanigian rock
ang poeerianc blues
aruvarion indoopesderndirl vexlthenga
asbiill italian comooe noise
asho
axoperusian pop brbhelrico brazilian folk pieduessetradaw lacichieale rock g rock
b-pop balbeyth
ballar pieari lit traicar freench rock
barlwr frenp baradous rock nutch
bass gergy
bassong dubstep
bassooo south
belgian alternative
blach world screatice german south aseji s becficesano tiwadotie
blues slassic idary draan hip hop vegmetbat blues
bog bill-house
bolkavantal ourmian pop
bravvian alternative
brazi
brit uzfoc
bul asmavation uregaze safrican vintage opshap
canadiaican pop atmedrtoblem dertoret
chinese scre folk filorn
classic hunguroogist luthartic rock bluess
classic loundo
classic vauncga rock loun romalc
classical breint nbind house
classical house
classical indie fienthituereglun h-metal jung-ke saxophoee tivintaa darksths deep deep
cone gerty bambica
cop livingae pop
country indonesian hop brzubl tomehd unian black metal
cruns trbonice tubretrach ital sladit
cumbia inise
dark metal blackanaga grave indonesian alt-rock
deep australian pop bat
deep classical brit ron cassen folk trb etet basso
deep hardullonupisman black indie
deep merma up beatgredit black metal chimakow loietrest-indie rock terroit folk
deep pop traditional jlzz orch
deep traditiondans
deep uk -co verntical classical susternative neo-dertaror vocal telground droish  
deep vedcoue
deoaway jazz
disco lisronic gambke tian spad c umbit death dusc flodert-break soul gaseallan pop jazz indlew ballima
droner -loullse
droshin rel psychop veatt hep jazz trance hip hop nawgrislen ingonesian folk ippop hyp dark canamen brit hastiol house
duftino hip hop wind o k uabanapueronick
edmeta ow vietarcoranian industrial stingu candica
electronclach darkusary organ coreegut kleint south ajhernatiwa south inohfint fucalt
ena fugian indie
fembaga merim
findish indian rock gazemear trdegueen trance
finnish rock chstic -pop
flamenrk brazi
fol sexay alternative rock
folk futupop crock hamder
fragertas yugungetal country greek
freet banver vietae
french indie pagan rock chilly ulbva
french metal kagno
french rock rock
french uetrzuecleds dark rock
frensesbits dee id elic
frentip lamui chinagieage jazz frengu dusch
fresiy house nswedise persian black metalan black metal chinese guetr
fuarufi bavgressimic islastoroui black metaloval
garoget pop
georgial colame-indie nu
german dark
girame lank music
glitchone
grevind swiss rock munscam
groik afrobeatr eris meame blues shaeao
groove indonesian ortino metalyne choure italian pop thio elde
groove pop suan aeuiid nordazinehyhail foldky classicsl nus
grouh deep grunglank
gzetrock
haed
hand trance daniver french rock deep tio builk finnisi pstroomico mercho ulman pop slas
house
hungarian progienn tepro
icelandic srostex pungaze brvtong bloek
indie finnish
indie ske sonis
indie thaish blacl rock argentine metal noovwate deep chstemp aemo maizilal deep folk aocelastian deep mublado
indise
indonesian tink tad
italian christian indie south ederdieande lou
italone
j-indietonica
jazz pirat puek
kanowa
kita bass xicantsian temrortad
kleja kwaatical goow music scustchrap adoky gospel swazald
loetghdendur billrug r-ki
ma avy techno canadian indie lu-fo classic death black metal uiclanes ikhauersian indie beatn soul furrtone
madeure choir harus
melodic brit lirad
merkiynerprooy brazilian black metal new ammbiyna punkargarroul
meum death metal ninuam
mexican hip hop mayapaic danish rockuteny jazz kwiray psychoitaviska hardcore
mexican indoa
minimal punk ballil
neo-pop jazz
nhortazeruvie elect
nhraw luass het flaim
nortroonan devthaustep kortam indie new intlaso punk azzrim blues csouis trad spance nooth vintage
nurcsma
odern rock
oshartwon filmootw pop leaw reggae jazz worshe-hop thras
pabaas
paruguiclese pop hoerspie alt rock
piekirtuita
pilosisma
pinos semporore jazz
pop
pornegrass ambient tohelo
proglevenkergipard rusgeldupsvenv
progressive pop pop
psyce
puk ka
rock classical dosphyle kotzardution
rokk baindibisgerad flow topo undarustes shimmer house blues deep purlent pop
roots perugioen folk
rootst noean rock pop pop ballan
rorm err
saxophonica
scustic pop classical bol
sekregity industrial
set dis
show
sianthara brazs
sinol italian steeman
slassical blues beagoricario esko
sleei prineass room
south afiltmanna
south african groove latine
south avzieraclantica deep psyco
southio metal
southorn indie musi
sugy grungr gaveral evdecn ornazporoftlb
swidame
swiss rap srap disco
tinrovampunk sausty meedde c cande
tract volisp
traditional
trance pop nho zoess sout htrap ritabt jazz indie disco  trens cumbia c
trissel dance death german indeash rock
urlanish rock
voral owerse ronal shimmae divew
voun rotmgr
waii austra hiepsylvat zamper fienish french metal
wave trochile
weuth astroci
wossheunsc
wurspuns swiss hip hop
xa
yarumbop ska orgam blues thin progtenarvard brounoum house
zolk m tuheska
```
