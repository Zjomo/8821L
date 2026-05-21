const fs = require('fs');
const path = require('path');
const zlib = require('zlib');

const srcPath = 'e:\\jupyter file\\2_Optics\\8821L\\final_merged_sorted_toc_20260521.docx';
const dstPath = 'c:\\Users\\Mr\\.trae-cn\\work\\6a0dbecbced5290c2425b1c2\\unpacked_doc';

// Simple ZIP extractor using only built-in modules
function extractZip(buffer, destDir) {
    let offset = 0;
    const entries = [];
    
    // Find end of central directory
    let eocdOffset = buffer.length - 22;
    while (eocdOffset >= 0) {
        if (buffer.readUInt32LE(eocdOffset) === 0x06054b50) break;
        eocdOffset--;
    }
    
    if (eocdOffset < 0) throw new Error('Invalid ZIP file');
    
    const centralDirOffset = buffer.readUInt32LE(eocdOffset + 16);
    const centralDirSize = buffer.readUInt32LE(eocdOffset + 12);
    const numEntries = buffer.readUInt16LE(eocdOffset + 8);
    
    // Read central directory
    let cdOffset = centralDirOffset;
    for (let i = 0; i < numEntries; i++) {
        if (buffer.readUInt32LE(cdOffset) !== 0x02014b50) break;
        
        const compressionMethod = buffer.readUInt16LE(cdOffset + 10);
        const compressedSize = buffer.readUInt32LE(cdOffset + 20);
        const uncompressedSize = buffer.readUInt32LE(cdOffset + 24);
        const filenameLength = buffer.readUInt16LE(cdOffset + 28);
        const extraLength = buffer.readUInt16LE(cdOffset + 30);
        const commentLength = buffer.readUInt16LE(cdOffset + 32);
        const localHeaderOffset = buffer.readUInt32LE(cdOffset + 42);
        
        const filename = buffer.toString('utf8', cdOffset + 46, cdOffset + 46 + filenameLength);
        
        entries.push({
            filename,
            compressionMethod,
            compressedSize,
            uncompressedSize,
            localHeaderOffset
        });
        
        cdOffset += 46 + filenameLength + extraLength + commentLength;
    }
    
    // Extract files
    const results = {};
    for (const entry of entries) {
        const localOffset = entry.localHeaderOffset;
        const localFilenameLength = buffer.readUInt16LE(localOffset + 26);
        const localExtraLength = buffer.readUInt16LE(localOffset + 28);
        const dataOffset = localOffset + 30 + localFilenameLength + localExtraLength;
        
        let data;
        if (entry.compressionMethod === 0) {
            // Stored (no compression)
            data = buffer.slice(dataOffset, dataOffset + entry.compressedSize);
        } else if (entry.compressionMethod === 8) {
            // Deflate
            const compressed = buffer.slice(dataOffset, dataOffset + entry.compressedSize);
            data = zlib.inflateSync(compressed);
        } else {
            continue;
        }
        
        results[entry.filename] = data;
        
        // Write to disk
        const fullPath = path.join(destDir, entry.filename);
        const dir = path.dirname(fullPath);
        fs.mkdirSync(dir, { recursive: true });
        fs.writeFileSync(fullPath, data);
    }
    
    return results;
}

console.log('正在解压Word文档...');
fs.mkdirSync(dstPath, { recursive: true });

const buffer = fs.readFileSync(srcPath);
const files = extractZip(buffer, dstPath);

console.log('解压完成!\n');

// List extracted files
console.log('='.repeat(60));
console.log('解压后的文件结构:');
console.log('='.repeat(60));

function listFiles(dir, indent = '') {
    const items = fs.readdirSync(dir);
    for (const item of items.sort()) {
        const fullPath = path.join(dir, item);
        const stat = fs.statSync(fullPath);
        if (stat.isDirectory()) {
            console.log(`${indent}${item}/`);
            listFiles(fullPath, indent + '  ');
        } else {
            console.log(`${indent}${item}`);
        }
    }
}

listFiles(dstPath);

// Analyze document.xml
console.log('\n' + '='.repeat(60));
console.log('分析 word/document.xml 中的标题结构');
console.log('='.repeat(60));

const docXmlPath = path.join(dstPath, 'word', 'document.xml');
if (fs.existsSync(docXmlPath)) {
    const content = fs.readFileSync(docXmlPath, 'utf8');
    
    // Find all heading styles
    const headings = [];
    const paragraphs = content.split('<w:p>');
    
    for (const para of paragraphs) {
        const styleMatch = para.match(/<w:pStyle w:val="([^"]+)"\/>/);
        if (styleMatch) {
            const style = styleMatch[1];
            if (style.includes('Heading') || style.includes('TOC')) {
                // Extract text
                const texts = [];
                const textMatches = para.matchAll(/<w:t[^>]*>([^<]*)<\/w:t>/g);
                for (const tm of textMatches) {
                    texts.push(tm[1]);
                }
                const text = texts.join('').substring(0, 100);
                
                // Extract font info
                const fontMatch = para.match(/<w:rFonts[^>]*>/);
                const sizeMatch = para.match(/<w:sz w:val="(\d+)"\/>/);
                const boldMatch = para.match(/<w:b\/>/);
                
                let props = [];
                if (fontMatch) {
                    const ascii = fontMatch[0].match(/w:ascii="([^"]+)"/);
                    const eastAsia = fontMatch[0].match(/w:eastAsia="([^"]+)"/);
                    if (ascii || eastAsia) {
                        props.push(`字体: ${ascii ? ascii[1] : ''} / ${eastAsia ? eastAsia[1] : ''}`);
                    }
                }
                if (sizeMatch) {
                    props.push(`字号: ${parseInt(sizeMatch[1]) / 2}pt`);
                }
                if (boldMatch) {
                    props.push('粗体');
                }
                
                headings.push({ style, text, props: props.length ? props : ['默认格式'] });
            }
        }
    }
    
    console.log(`\n找到 ${headings.length} 个标题/TOC段落:\n`);
    
    // Count by style
    const styleCounts = {};
    for (const h of headings) {
        styleCounts[h.style] = (styleCounts[h.style] || 0) + 1;
    }
    
    console.log('样式统计:');
    console.log('-'.repeat(40));
    for (const [style, count] of Object.entries(styleCounts).sort()) {
        console.log(`  ${style}: ${count} 个`);
    }
    
    console.log('\n\n详细标题列表:');
    console.log('-'.repeat(60));
    for (let i = 0; i < Math.min(50, headings.length); i++) {
        const h = headings[i];
        console.log(`\n${i + 1}. 样式: ${h.style}`);
        console.log(`   文本: ${h.text}`);
        console.log(`   格式: ${h.props.join(', ')}`);
    }
    
    if (headings.length > 50) {
        console.log(`\n... 还有 ${headings.length - 50} 个标题未显示`);
    }
}

// Analyze styles.xml
console.log('\n\n' + '='.repeat(60));
console.log('分析 word/styles.xml 中的样式定义');
console.log('='.repeat(60));

const stylesXmlPath = path.join(dstPath, 'word', 'styles.xml');
if (fs.existsSync(stylesXmlPath)) {
    const content = fs.readFileSync(stylesXmlPath, 'utf8');
    
    // Find heading style definitions
    const styleBlocks = content.split('<w:style ');
    const headingStyles = [];
    
    for (const block of styleBlocks) {
        const idMatch = block.match(/w:styleId="([^"]+)"/);
        if (idMatch) {
            const styleId = idMatch[1];
            if (styleId.includes('Heading') || styleId.includes('TOC')) {
                const nameMatch = block.match(/<w:name w:val="([^"]+)"\/>/);
                const basedOnMatch = block.match(/<w:basedOn w:val="([^"]+)"\/>/);
                
                // Extract outline level
                const outlineMatch = block.match(/<w:outlineLvl w:val="(\d+)"\/>/);
                
                // Extract font size
                const sizeMatch = block.match(/<w:sz w:val="(\d+)"\/>/);
                
                // Extract font
                const fontMatch = block.match(/<w:rFonts[^>]*>/);
                
                let props = [];
                if (outlineMatch) {
                    props.push(`大纲级别: ${outlineMatch[1]}`);
                }
                if (fontMatch) {
                    const ascii = fontMatch[0].match(/w:ascii="([^"]+)"/);
                    const eastAsia = fontMatch[0].match(/w:eastAsia="([^"]+)"/);
                    if (ascii || eastAsia) {
                        props.push(`字体: ${ascii ? ascii[1] : ''} / ${eastAsia ? eastAsia[1] : ''}`);
                    }
                }
                if (sizeMatch) {
                    props.push(`字号: ${parseInt(sizeMatch[1]) / 2}pt`);
                }
                
                headingStyles.push({
                    id: styleId,
                    name: nameMatch ? nameMatch[1] : styleId,
                    basedOn: basedOnMatch ? basedOnMatch[1] : null,
                    props
                });
            }
        }
    }
    
    console.log(`\n找到 ${headingStyles.length} 个标题相关样式定义:\n`);
    for (const s of headingStyles) {
        console.log(`样式ID: ${s.id}`);
        console.log(`  名称: ${s.name}`);
        if (s.basedOn) {
            console.log(`  基于: ${s.basedOn}`);
        }
        if (s.props.length) {
            console.log(`  属性: ${s.props.join(', ')}`);
        }
        console.log();
    }
}

console.log('\n分析完成!');
